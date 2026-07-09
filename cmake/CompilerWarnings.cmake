# Compiler warning helpers for the ORE project.
#
# Provides two functions:
#   ore_set_warnings(<target>)             - strict warnings + warnings-as-errors
#   ore_set_warnings_no_werror(<target>)   - strict warnings, no -Werror/-WX
#
# The relaxed form is used for the test executable so that internal warnings
# in GoogleTest's headers cannot break our build.

function(ore_set_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /W4          # High warning level
            /permissive- # Strict standards conformance
            /WX          # Warnings as errors
        )
    else()
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            -Werror
            -Wshadow
            -Wnon-virtual-dtor
            -Wold-style-cast
            -Wcast-align
            -Wunused
            -Woverloaded-virtual
            -Wdouble-promotion
        )
    endif()
endfunction()

function(ore_set_warnings_no_werror target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else()
        target_compile_options(${target} PRIVATE
            -Wall -Wextra -Wpedantic
            -Wshadow -Wnon-virtual-dtor -Wold-style-cast
            -Wcast-align -Wunused -Woverloaded-virtual -Wdouble-promotion
        )
    endif()
endfunction()
