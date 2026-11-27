#!/bin/sh
# Packs dist/Mira.app (PyInstaller) into dist/installer/Mira-<version>-macos-arm64.pkg.
#
#   packaging/macos/build_pkg.sh 1.2.3
#
# The package always has the identifier io.github.w1nte.mira, so the macOS Installer offers
# "Upgrade" over an installed version. Its scripts quit a running Mira first (preinstall) and
# migrate the library's database afterwards (postinstall). It is not signed (see docs/installation.md).
set -eu
version=${1:?usage: build_pkg.sh VERSION}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
app=$root/dist/Mira.app
work=$root/build/macos-pkg
out=$root/dist/installer
identifier=io.github.w1nte.mira
buddy=/usr/libexec/PlistBuddy

[ -d "$app" ] || { echo "build_pkg.sh: $app is missing (run PyInstaller first)" >&2; exit 1; }
exe=$("$buddy" -c "Print :CFBundleExecutable" "$app/Contents/Info.plist")
if [ "$exe" != Mira ]; then
  echo "build_pkg.sh: the bundle would start $exe instead of Mira" >&2
  exit 1
fi

rm -rf "$work"
mkdir -p "$work/root" "$work/scripts" "$out"
ditto "$app" "$work/root/Mira.app" # keeps symlinks, permissions and the ad-hoc signatures
cp "$here/scripts/preinstall" "$here/scripts/postinstall" "$work/scripts/"
chmod 755 "$work/scripts/preinstall" "$work/scripts/postinstall"

# Every bundle in the payload: install exactly into /Applications (never "relocate" an update to
# a copy of the app found elsewhere), and always replace it, also with the same or an older
# version (a reinstall, a pre-release followed by its release).
plist=$work/component.plist
pkgbuild --analyze --root "$work/root" "$plist"
i=0
while "$buddy" -c "Print :$i" "$plist" >/dev/null 2>&1; do
  "$buddy" -c "Set :$i:BundleIsRelocatable false" -c "Set :$i:BundleIsVersionChecked false" \
    "$plist"
  i=$((i + 1))
done
[ "$i" -gt 0 ] || { echo "build_pkg.sh: pkgbuild found no bundle" >&2; exit 1; }

pkgbuild --root "$work/root" --component-plist "$plist" --identifier "$identifier" \
  --version "$version" --install-location /Applications --scripts "$work/scripts" \
  "$work/Mira.pkg"

sed "s/@VERSION@/$version/g" "$here/distribution.xml" >"$work/distribution.xml"
productbuild --distribution "$work/distribution.xml" --resources "$here/resources" \
  --package-path "$work" "$out/Mira-$version-macos-arm64.pkg"
pkgutil --check-signature "$out/Mira-$version-macos-arm64.pkg" || true # "no signature"
ls -l "$out"
