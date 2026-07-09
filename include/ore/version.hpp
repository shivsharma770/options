/**
 * @file version.hpp
 * @brief Compile-time and run-time version identification for the library.
 */
#pragma once

namespace ore {

/**
 * Semantic version of the library as a null-terminated string
 * (`"MAJOR.MINOR.PATCH"`). Kept as a free function (rather than a macro)
 * so it is namespaced and can be mocked or overridden in tests if needed.
 */
[[nodiscard]] const char* version() noexcept;

} // namespace ore
