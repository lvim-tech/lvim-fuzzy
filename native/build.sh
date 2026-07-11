#!/bin/sh
# Build the lvim-fuzzy native matcher into native/build/liblvimfuzzy.so — the
# path the Lua FFI loader (lua/lvim-fuzzy/init.lua) probes first. Requires a Rust
# toolchain (cargo); if none is present the plugin transparently uses its
# pure-Lua fallback, so this build is optional.
#
#   sh native/build.sh
set -e
cd "$(dirname "$0")"

cargo build --release

mkdir -p build
# shared-library extension differs per OS (.so Linux, .dylib macOS, .dll Windows)
for ext in so dylib dll; do
    if [ -f "target/release/liblvimfuzzy.$ext" ]; then
        cp -f "target/release/liblvimfuzzy.$ext" "build/liblvimfuzzy.$ext"
        echo "installed build/liblvimfuzzy.$ext"
    fi
done
