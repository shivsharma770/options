/**
 * @file test_research_engine.cpp
 * @brief `HistoricalResearchEngine` iteration, progress callback,
 *        output-dir handling, error trapping, and parallel-equals-
 *        serial determinism.
 */
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <ore/marketdata/historical_dataset.hpp>
#include <ore/research/historical_research_engine.hpp>
#include <ore/research/research_context.hpp>
#include <ore/research/research_report.hpp>
#include <ore/research/research_study.hpp>

#include "research_test_helpers.hpp"

namespace {

using namespace ore::research;
using namespace ore::marketdata;
namespace fs = std::filesystem;

/// A minimal study that just counts snapshots it sees. Enough to
/// exercise the engine loop without depending on any pricing code.
class CountingStudy final : public ResearchStudy {
public:
    [[nodiscard]] std::string_view name() const noexcept override {
        return "CountingStudy";
    }

    void begin(const ResearchContext&) override { begins_++; }
    void process(const ResearchContext& ctx) override {
        std::lock_guard<std::mutex> lock(m_);
        dates_.push_back(ctx.date());
    }
    void end(const ResearchContext&, ResearchReport& r) override {
        ends_++;
        r.processed_contracts = dates_.size();
    }

    [[nodiscard]] std::unique_ptr<ResearchStudy> clone() const override {
        return std::make_unique<CountingStudy>();
    }
    void merge(const ResearchStudy& other) override {
        const auto* rhs = dynamic_cast<const CountingStudy*>(&other);
        if (!rhs) return;
        std::lock_guard<std::mutex> lock(m_);
        dates_.insert(dates_.end(), rhs->dates_.begin(), rhs->dates_.end());
    }

    const std::vector<std::chrono::year_month_day>& dates() const noexcept {
        return dates_;
    }
    std::size_t begins() const noexcept { return begins_; }
    std::size_t ends() const noexcept { return ends_; }

private:
    std::vector<std::chrono::year_month_day> dates_{};
    std::size_t begins_{0};
    std::size_t ends_{0};
    mutable std::mutex m_{};
};

fs::path scratch_dir(std::string_view label) {
    auto p = fs::temp_directory_path() /
        ("ore_research_" + std::string{label} + "_" +
         std::to_string(std::hash<std::string_view>{}(label)));
    fs::remove_all(p);
    fs::create_directories(p);
    return p;
}

TEST(ResearchEngine, ProcessesEveryDayInOrder) {
    auto ds = ore::research::testing::make_synthetic_dataset(4);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch_dir("in_order");
    HistoricalResearchEngine engine{ds, cfg};

    CountingStudy s{};
    auto report = engine.run(s);

    EXPECT_EQ(report.processed_days, 4u);
    ASSERT_EQ(s.dates().size(), 4u);
    for (std::size_t i = 0; i + 1 < s.dates().size(); ++i) {
        EXPECT_LT(std::chrono::sys_days{s.dates()[i]},
                  std::chrono::sys_days{s.dates()[i + 1]});
    }
    EXPECT_EQ(s.begins(), 1u);
    EXPECT_EQ(s.ends(),   1u);
}

TEST(ResearchEngine, HandlesEmptyDataset) {
    HistoricalDataset empty{"SPY", {}};
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch_dir("empty");
    HistoricalResearchEngine engine{empty, cfg};
    CountingStudy s{};
    auto report = engine.run(s);
    EXPECT_EQ(report.processed_days, 0u);
    EXPECT_EQ(s.begins(), 0u);
    EXPECT_EQ(s.ends(),   0u);
}

TEST(ResearchEngine, HandlesSingleDayDataset) {
    auto ds = ore::research::testing::make_synthetic_dataset(1);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch_dir("one_day");
    HistoricalResearchEngine engine{ds, cfg};
    CountingStudy s{};
    auto report = engine.run(s);
    EXPECT_EQ(report.processed_days, 1u);
    EXPECT_EQ(s.begins(), 1u);
    EXPECT_EQ(s.ends(),   1u);
}

TEST(ResearchEngine, ProgressCallbackReceivesEveryDay) {
    auto ds = ore::research::testing::make_synthetic_dataset(6);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch_dir("progress");
    std::vector<std::pair<std::size_t, std::size_t>> calls;
    cfg.progress = [&](std::size_t done, std::size_t total) {
        calls.emplace_back(done, total);
    };
    HistoricalResearchEngine engine{ds, cfg};
    CountingStudy s{};
    (void)engine.run(s);

    ASSERT_EQ(calls.size(), 6u);
    for (std::size_t i = 0; i < calls.size(); ++i) {
        EXPECT_EQ(calls[i].second, 6u);
        EXPECT_EQ(calls[i].first,  i + 1u); // 1-based done count, in order
    }
}

TEST(ResearchEngine, CreatesOutputDirectory) {
    auto ds = ore::research::testing::make_synthetic_dataset(1);
    auto root = scratch_dir("mkdir");
    auto target = root / "nested" / "subdir";
    fs::remove_all(target);
    ASSERT_FALSE(fs::exists(target));

    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = target;
    HistoricalResearchEngine engine{ds, cfg};
    CountingStudy s{};
    (void)engine.run(s);
    EXPECT_TRUE(fs::exists(target));
    EXPECT_TRUE(fs::is_directory(target));
}

TEST(ResearchEngine, ContinueOnErrorSurfacesAsReportEntries) {
    // Study that always throws — the engine should catch and record.
    class Blowing final : public ResearchStudy {
    public:
        std::string_view name() const noexcept override { return "Blowing"; }
        void process(const ResearchContext&) override {
            throw std::runtime_error("boom");
        }
    };
    auto ds = ore::research::testing::make_synthetic_dataset(3);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch_dir("throw");
    cfg.continue_on_error = true;
    HistoricalResearchEngine engine{ds, cfg};
    Blowing s{};
    auto report = engine.run(s);
    EXPECT_EQ(report.errors.size(), 3u);
    EXPECT_EQ(report.processed_days, 3u);
}

TEST(ResearchEngine, AcceptsTemporaryStudy) {
    auto ds = ore::research::testing::make_synthetic_dataset(2);
    HistoricalResearchEngine::Config cfg{};
    cfg.output_dir = scratch_dir("temp_study");
    HistoricalResearchEngine engine{ds, cfg};
    // The milestone spec explicitly requires engine.run(Study{}).
    auto report = engine.run(CountingStudy{});
    EXPECT_EQ(report.processed_days, 2u);
}

TEST(ResearchEngine, ParallelReproducesSerialDates) {
    auto ds = ore::research::testing::make_synthetic_dataset(20);

    // Serial baseline.
    HistoricalResearchEngine::Config serial_cfg{};
    serial_cfg.threads    = 1;
    serial_cfg.output_dir = scratch_dir("par_serial");
    HistoricalResearchEngine serial_engine{ds, serial_cfg};
    CountingStudy serial_s{};
    (void)serial_engine.run(serial_s);

    // Parallel with as many threads as we can get.
    HistoricalResearchEngine::Config par_cfg{};
    par_cfg.threads = std::max(2u, std::thread::hardware_concurrency());
    par_cfg.output_dir = scratch_dir("par_parallel");
    HistoricalResearchEngine par_engine{ds, par_cfg};
    CountingStudy par_s{};
    (void)par_engine.run(par_s);

    // Same date set (order-independent).
    auto serial_dates = serial_s.dates();
    auto par_dates    = par_s.dates();
    std::sort(serial_dates.begin(), serial_dates.end(),
              [](auto a, auto b) {
                  return std::chrono::sys_days{a} < std::chrono::sys_days{b};
              });
    std::sort(par_dates.begin(), par_dates.end(),
              [](auto a, auto b) {
                  return std::chrono::sys_days{a} < std::chrono::sys_days{b};
              });
    EXPECT_EQ(serial_dates, par_dates);
    EXPECT_EQ(par_dates.size(), 20u);
}

TEST(ResearchEngine, NonCloneableStudyFallsBackToSerial) {
    // A study that refuses to clone should still run correctly when
    // the caller asks for many threads.
    class NoClone final : public ResearchStudy {
    public:
        std::string_view name() const noexcept override { return "NoClone"; }
        void process(const ResearchContext&) override { ++count; }
        std::size_t count{0};
    };

    auto ds = ore::research::testing::make_synthetic_dataset(8);
    HistoricalResearchEngine::Config cfg{};
    cfg.threads    = 8;
    cfg.output_dir = scratch_dir("no_clone");
    HistoricalResearchEngine engine{ds, cfg};
    NoClone s{};
    (void)engine.run(s);
    EXPECT_EQ(s.count, 8u);
}

} // namespace
