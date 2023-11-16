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
#include <sqlite3.h>
#include <cstdio>
#include <fstream>
#include <vector>
#include "nvblox/utils/logging.h"

#include "nvblox/serialization/internal/sqlite_database.h"

namespace nvblox {

SqliteDatabase::~SqliteDatabase() {
  if (db_ != nullptr) {
    close();
  }
}

/// Is the current file valid?
bool SqliteDatabase::valid() const {
  // TODO: check actual status from sqlite side?
  return db_ != nullptr;
}

/// Open a file for reading (std::ios::in) or writing (std::ios::out) or both.
bool SqliteDatabase::open(const std::string& filename,
                          std::ios_base::openmode openmode) {
  int flags = 0;
  // We always open stuff for reading no matter what. :)
  if (openmode & std::ios::out) {
    flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;
  } else if (openmode & std::ios::in) {
    flags = SQLITE_OPEN_READONLY;
  }
  // If the file exists and we've requested truncation...
  if (openmode & std::ios::trunc && std::ifstream(filename.c_str())) {
    std::remove(filename.c_str());
  }

  int status = sqlite3_open_v2(filename.c_str(), &db_, flags, nullptr);
  if (status != 0) {
    LOG(ERROR) << "Can't open database: " << sqlite3_errmsg(db_);
    close();
  }
  return status == SQLITE_OK;
}

/// Close the file.
bool SqliteDatabase::close() {
  int status = sqlite3_close_v2(db_);
  db_ = nullptr;
  return status == SQLITE_OK;
}

bool SqliteDatabase::runStatement(const std::string& sql_query) {
  // Now run the stuff.
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_)
               << "\nQuery: " << sql_query;
    return false;
  }
  if (sqlite3_step(statement) != SQLITE_DONE) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runStatementWithBlob(const std::string& sql_query,
                                          const std::vector<Byte>& blob) {
  // Now run the stuff.
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_)
               << "\nQuery: " << sql_query;
    return false;
  }

  // Bind the blob. Indexing starts at 1. Dunno why.
  sqlite3_bind_blob(statement, 1, blob.data(), blob.size(), SQLITE_STATIC);

  if (sqlite3_step(statement) != SQLITE_DONE) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runSingleQueryString(const std::string& sql_query,
                                          std::string* result) {
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_);
    return false;
  }
  if (sqlite3_step(statement) != SQLITE_ROW) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  *result = reinterpret_cast<const char*>(sqlite3_column_text(statement, 0));
  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runSingleQueryInt(const std::string& sql_query,
                                       int* result) {
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_);
    return false;
  }
  if (sqlite3_step(statement) != SQLITE_ROW) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  *result = sqlite3_column_int(statement, 0);
  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runSingleQueryFloat(const std::string& sql_query,
                                         float* result) {
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_);
    return false;
  }
  if (sqlite3_step(statement) != SQLITE_ROW) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  *result = static_cast<float>(sqlite3_column_double(statement, 0));
  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runSingleQueryBlob(const std::string& sql_query,
                                        std::vector<Byte>* result) {
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_);
    return false;
  }
  if (sqlite3_step(statement) != SQLITE_ROW) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }

  const void* blob = sqlite3_column_blob(statement, 0);
  size_t blob_size = sqlite3_column_bytes(statement, 0);
  result->resize(blob_size);
  memcpy(result->data(), blob, blob_size);

  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runMultipleQueryString(const std::string& sql_query,
                                            std::vector<std::string>* result) {
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_);
    return false;
  }
  int rc = sqlite3_step(statement);
  while (rc == SQLITE_ROW) {
    result->emplace_back(
        reinterpret_cast<const char*>(sqlite3_column_text(statement, 0)));

    rc = sqlite3_step(statement);
  }
  if (rc != SQLITE_DONE) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  sqlite3_finalize(statement);
  return retval;
}

bool SqliteDatabase::runMultipleQueryIndex3D(const std::string& sql_query,
                                             std::vector<Index3D>* result) {
  bool retval = true;
  sqlite3_stmt* statement;
  int status = sqlite3_prepare_v2(db_, sql_query.c_str(), -1, &statement, 0);
  if (status != SQLITE_OK) {
    LOG(ERROR) << "Preparing query failed: " << sqlite3_errmsg(db_);
    return false;
  }
  int rc = sqlite3_step(statement);
  while (rc == SQLITE_ROW) {
    Index3D index(static_cast<float>(sqlite3_column_double(statement, 0)),
                  static_cast<float>(sqlite3_column_double(statement, 1)),
                  static_cast<float>(sqlite3_column_double(statement, 2)));
    result->emplace_back(index);
    rc = sqlite3_step(statement);
  }
  if (rc != SQLITE_DONE) {
    LOG(ERROR) << "Query execution failed: " << sqlite3_errmsg(db_);
    retval = false;
  }
  sqlite3_finalize(statement);
  return retval;
}

}  // namespace nvblox
