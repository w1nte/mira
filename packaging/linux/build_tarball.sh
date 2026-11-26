#!/bin/sh
# Packs the PyInstaller folder (dist/vision-tagger) and install.sh into
# dist/installer/Mira-<version>-linux-x86_64-<flavor>.tar.gz.
#
#   packaging/linux/build_tarball.sh 1.2.3 cpu
#
# A GitHub release file can be at most 2 GiB: a bigger archive (the gpu build, with CUDA) is
# split into .part00, .part01, ... files; `cat Mira-...tar.gz.part* | tar -xz` unpacks them.
set -eu
version=${1:?usage: build_tarball.sh VERSION FLAVOR}
flavor=${2:?usage: build_tarball.sh VERSION FLAVOR}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
name=Mira-$version-linux-x86_64-$flavor
work=$root/build/linux
out=$root/dist/installer
max_mib=1900 # per file, safely below the limit

[ -x "$root/dist/vision-tagger/vision-tagger-cli" ] || {
  echo "build_tarball.sh: dist/vision-tagger is missing (run PyInstaller first)" >&2
  exit 1
}
rm -rf "$work"
mkdir -p "$work/$name" "$out"
cp "$here/install.sh" "$work/$name/install.sh"
chmod 755 "$work/$name/install.sh"
cp "$root/src/vision_tagger/web/static/icons/icon-512.png" "$work/$name/mira.png"
printf '%s\n' "$version" >"$work/$name/VERSION"
cat >"$work/$name/README.txt" <<EOF
Mira $version for Linux (x86_64, $flavor)

Install, or update an installed Mira (your library is kept and its database migrated):

    ./install.sh                 # into ~/.local/share/mira, launchers in ~/.local/bin
    ./install.sh --prefix DIR    # somewhere else
    ./install.sh --uninstall     # remove the app; the library is kept

Then start Mira from the applications menu or with \`mira\`. More: https://github.com/w1nte/vision-tagger
EOF

# the app folder goes in as $name/app without a copy (the gpu build has gigabytes)
archive=$out/$name.tar.gz
tar -czf "$archive" -C "$work" "$name" \
  -C "$root/dist" --transform "flags=r;s|^vision-tagger|$name/app|" vision-tagger
size=$(wc -c <"$archive")
if [ "$size" -gt $((max_mib * 1024 * 1024)) ]; then
  split -b "${max_mib}M" -d -a 2 "$archive" "$archive.part"
  rm "$archive"
  echo "split into parts: cat $name.tar.gz.part* | tar -xz"
fi
ls -l "$out"
