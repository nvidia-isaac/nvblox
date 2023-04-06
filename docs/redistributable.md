# Redistributable Building 
This will walk you through how to build a single libnvblox shared library with only static dependencies. The dependencies must be built manually with -fPIC (position independent code) and linked as static libraries.  
You generally shouldn't need to do this except for building nvblox to be included in Bazel for example; the exported targets should work fine for inclusion in CMake projects.

## Step 1: Install dependencies
**Sqlite3**:  
Download here: https://www.sqlite.org/2022/sqlite-autoconf-3390400.tar.gz
 
Unzip & build using this command:
```
CFLAGS=-fPIC ./configure --prefix=/home/USERNAME/code/external/sqlite-autoconf-3390400/install/
```

**glog**:  
Download here: https://github.com/google/glog/archive/refs/tags/v0.4.0.zip

Unzip & build using this command:
```
mkdir build install
cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DCMAKE_INSTALL_PREFIX=/home/USERNAME/code/external/glog-0.4.0/install/ \
-DWITH_GFLAGS=OFF -DBUILD_SHARED_LIBS=OFF \
&& make -j8 && make install
```

**gflags**:  
Download here: https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.zip

Unzip & build using this command:
```
mkdir build install
cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
-DCMAKE_INSTALL_PREFIX=/home/USERNAME/code/external/gflags-2.2.2/install/ \
-DGFLAGS_BUILD_STATIC_LIBS=ON -DGFLAGS=google \
&& make -j8 && make install
```

## Step 2: Build nvblox
Build nvblox with the paths set to where you've unzipped and built the dependencies:
```
cd nvblox/
mkdir build install
cmake .. -DCMAKE_INSTALL_PREFIX=/home/helen/code/nvblox/nvblox/install/ \
-DBUILD_FOR_ALL_ARCHS=TRUE -DBUILD_REDISTRIBUTABLE=TRUE \
-DSQLITE3_BASE_PATH="/home/USERNAME/code/external/sqlite-autoconf-3390400/install/" \ 
-DGLOG_BASE_PATH="/home/USERNAME/code/external/glog-0.4.0/install/" \
-DGFLAGS_BASE_PATH="/home/USERNAME/code/external/gflags-2.2.2/install/" \
&& make -j8 && make install
```