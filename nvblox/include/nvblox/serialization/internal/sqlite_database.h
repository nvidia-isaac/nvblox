/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include <vector>

#include "nvblox/core/types.h"

// forward declaration
struct sqlite3;

namespace nvblox {
/// Class to wrap access to the C interface of SQLite in a slightly more
/// usable format.
class SqliteDatabase {
 public:
  /// Default constructor for invalid file. Must call open before using.
  SqliteDatabase() = default;
  virtual ~SqliteDatabase();

  /// Is the current file valid? Basically, have we opened a file that sqlite
  /// agrees is a database?
  bool valid() const;

  /// Open a file for reading (std::ios::in) or writing (std::ios::out) or both.
  bool open(const std::string& filename,
            std::ios_base::openmode openmode = std::ios::in);

  /// Close the file.
  bool close();

  /// Run a statement that does not have a return value.
  bool runStatement(const std::string& statement);
  /// Run a return-value-less statement on a byte blob.
  bool runStatementWithBlob(const std::string& statement,
                            const std::vector<Byte>& blob);

  /// Run a query that has a SINGLE return value of the given type:
  bool runSingleQueryString(const std::string& sql_query, std::string* result);
  bool runSingleQueryInt(const std::string& sql_query, int* result);
  bool runSingleQueryFloat(const std::string& sql_query, float* result);
  bool runSingleQueryBlob(const std::string& sql_query,
                          std::vector<Byte>* result);

  /// Returns MULTIPLE values.
  bool runMultipleQueryString(const std::string& sql_query,
                              std::vector<std::string>* result);
  /// Return multiple values in an index.
  bool runMultipleQueryIndex3D(const std::string& sql_query,
                               std::vector<Index3D>* result);

 private:
  sqlite3* db_ = nullptr;
};

}  // namespace nvblox
