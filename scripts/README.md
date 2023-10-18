# Scripts
This folder contains scripts we use.

## `make_deployable_artifact.sh`
Internally we need to deploy nvblox as a minimum dependency library (`.so`) into other systems. To do this we statically link nvblox dependencies.

This script (`make_deployable_artifact.sh`) builds this artifact. However, it requires that the appropriate dependencies are built before running and exist in a particular folder structure (automating this is left as an exercise to the reader :wink:).

The folder structure should look like:

```bash
├── nvblox/
├── glog-0.4.0/
├── sqlite-autoconf-3390400/
├── gflags-2.2.2/
```

Once you have the dependencies built in this structure (see below), run the script, and libs and headers are installed to `nvblox/nvblox/install`.

### Build the dependencies
We need `glog`, `sqlite3`, and `gflags` prepared for static linking. To get these ready do:

#### sqlite3

```bash
wget https://sqlite.org/2022/sqlite-autoconf-3400100.tar.gz
tar -xvf sqlite-autoconf-3400100.tar.gz
cd sqlite-autoconf-3400100
mkdir install
CFLAGS=-fPIC ./configure --prefix=$PWD/install/
make
make install
```

#### glog
```bash
wget https://github.com/google/glog/archive/refs/tags/v0.4.0.tar.gz
tar -xvf v0.4.0.tar.gz
cd glog-0.4.0
mkdir build
mkdir install
cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$PWD/../install/ -DWITH_GFLAGS=OFF -DBUILD_SHARED_LIBS=OFF
make -j8
make install
```

#### gflags
```bash
wget https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.tar.gz
tar -xvf v2.2.2.tar.gz
cd gflags-2.2.2/
mkdir build
mkdir install
cd build
cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=$PWD/../install/ -DGFLAGS_BUILD_STATIC_LIBS=ON -DGFLAGS=google && make -j8 && make install
```
