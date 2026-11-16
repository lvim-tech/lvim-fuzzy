#!/bin/sh
# Build the lvim-fuzzy native library into native/build/liblvimfuzzy.so — the
# path the Lua FFI loader (lua/lvim-fuzzy/init.lua) probes first. Requires a Rust
# toolchain (cargo) and, on the first build, the network: cargo fetches the
# search engine (lvim-fuzzy-core) from GitHub. The library is required — there
# is no pure-Lua matcher.
#
#   sh native/build.sh
#
# The compile cache goes to $CARGO_TARGET_DIR when it is set (lvim-pack sets it to
# a cache directory outside the plugin), else to native/target.
set -e
cd "$(dirname "$0")"

cargo build --release
target="${CARGO_TARGET_DIR:-target}"

mkdir -p build
# shared-library extension differs per OS (.so Linux, .dylib macOS, .dll Windows)
for ext in so dylib dll; do
    if [ -f "$target/release/liblvimfuzzy.$ext" ]; then
        cp -f "$target/release/liblvimfuzzy.$ext" "build/liblvimfuzzy.$ext"
        echo "installed build/liblvimfuzzy.$ext"
    fi
done
