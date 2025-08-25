if (USE_SYSTEM_SQLITE3)
  find_package(sqlite3 REQUIRED)
else()
include(FetchContent)

# Download the amalgamation version of sqlite that contains the source code in
# one massive source file.
FetchContent_Declare(
  ext_sqlite3
  OVERRIDE_FIND_PACKAGE
  SYSTEM
  URL https://sqlite.org/2025/sqlite-amalgamation-3500400.zip
  URL_HASH MD5=440abd85c5ee3297dd388ade51fec0cc)

FetchContent_MakeAvailable(ext_sqlite3)
add_library(sqlite3 ${ext_sqlite3_SOURCE_DIR}/sqlite3.c)
target_include_directories(sqlite3 PUBLIC ${ext_sqlite3_SOURCE_DIR})

# Apply nvblox compile options to exported targets
set_nvblox_compiler_options_nowarnings(sqlite3)
endif()
