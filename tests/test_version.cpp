#include <gtest/gtest.h>

#include <ore/version.hpp>

#include <cstring>
#include <string_view>

TEST(VersionTest, ReturnsNonEmptyString) {
    const char* v = ore::version();
    ASSERT_NE(v, nullptr);
    EXPECT_GT(std::strlen(v), 0U);
}

TEST(VersionTest, IsSemverLike) {
    const std::string_view v = ore::version();
    EXPECT_NE(v.find('.'), std::string_view::npos)
        << "Expected semantic version format 'MAJOR.MINOR.PATCH', got '" << v << "'";
}
