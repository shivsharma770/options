#include <ore/version.hpp>

namespace ore {

const char* version() noexcept {
    // Kept in sync with `project(... VERSION ...)` in the top-level
    // CMakeLists.txt. Future improvement: generate this from a
    // `configure_file` template driven by ${PROJECT_VERSION}.
    return "0.1.0";
}

} // namespace ore
