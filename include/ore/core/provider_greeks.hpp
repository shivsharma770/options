/**
 * @file provider_greeks.hpp
 * @brief Sensitivities as reported by a market-data provider.
 *
 * Design notes
 * ------------
 * - `ProviderGreeks` is the **raw** Greek values as they arrive from a
 *   vendor's end-of-day feed. It is *not* the same type as
 *   `ore::pricing::Greeks` deliberately:
 *
 *     - `ore::pricing::Greeks` is the output of a pricing engine, always
 *       populated, in the project's canonical unit convention.
 *     - `ore::core::ProviderGreeks` may have any subset of its fields
 *       populated (vendors routinely omit rho or theta for deep OTM
 *       contracts) and may use whatever unit convention the vendor
 *       chose. Callers who want to compare against
 *       `ore::pricing::Greeks` are responsible for the unit conversion.
 *
 * - Keeping this type in `ore::core` avoids a circular dependency:
 *   `ore::core::Quote` needs to carry these values, but pulling
 *   `ore::pricing::Greeks` into `ore::core` would invert the module
 *   hierarchy. A parallel POD is the correct decoupling.
 *
 * - `std::optional<double>` is used deliberately for every field so a
 *   published-but-zero value (`std::optional{0.0}`) stays unambiguously
 *   distinct from a missing value (`std::nullopt`). Research studies
 *   that skip contracts with missing Greeks rely on that distinction.
 *
 * ### Unit convention (as stored)
 * Whatever the vendor supplied. The `EodOptionsLoader` for
 * OptionsDX/Delta-Neutral archives preserves the raw column values:
 *   - `delta`, `gamma`, `vega`, `theta`, `rho` are stored as-parsed.
 *   - Vendors typically publish "per-1%" vega and "per-day" theta;
 *     the project's canonical convention is "per-unit-vol" vega and
 *     "per-year" theta. Consumers must convert when comparing to
 *     `ore::pricing::Greeks`. `docs/HISTORICAL_RESEARCH.md`
 *     documents the specific conversion this project applies.
 */
#pragma once

#include <optional>

namespace ore::core {

/**
 * @brief Vendor-supplied Greeks attached to a `Quote`.
 *
 * Every field is optional. Vendors routinely omit rho on short-dated
 * contracts and theta on illiquid strikes; the study code treats a
 * `nullopt` field as "not reported" and skips the row instead of
 * substituting a default.
 */
struct ProviderGreeks {
    /** \f$ \Delta \f$ as reported by the vendor. */
    std::optional<double> delta{};
    /** \f$ \Gamma \f$ as reported by the vendor. */
    std::optional<double> gamma{};
    /** \f$ \Theta \f$ as reported by the vendor. Vendor unit convention
     *  (frequently per-day) is preserved verbatim. */
    std::optional<double> theta{};
    /** \f$ \mathcal{V} \f$ as reported by the vendor. Vendor unit
     *  convention (frequently per-1%-vol) is preserved verbatim. */
    std::optional<double> vega{};
    /** \f$ \rho \f$ as reported by the vendor. */
    std::optional<double> rho{};

    /** True iff every field is populated. */
    [[nodiscard]] constexpr bool complete() const noexcept {
        return delta.has_value() && gamma.has_value() && theta.has_value()
            && vega.has_value()  && rho.has_value();
    }

    /** True iff at least one field is populated. */
    [[nodiscard]] constexpr bool any() const noexcept {
        return delta.has_value() || gamma.has_value() || theta.has_value()
            || vega.has_value()  || rho.has_value();
    }

    friend bool operator==(const ProviderGreeks&, const ProviderGreeks&) = default;
};

} // namespace ore::core
