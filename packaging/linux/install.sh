#!/bin/sh
# Installs Mira for the current user (no root needed), or updates an installed Mira.
#
#   ./install.sh                  install or update (into ~/.local/share/mira)
#   ./install.sh --prefix DIR     install or update in DIR instead
#   ./install.sh --uninstall      remove the app and its launchers; your library is kept
#
# An update first quits a running Mira (a running pipeline is paused and continues on the next
# start), copies the new version next to the old one and then swaps them, and finally migrates
# the library's database (a backup copy is made first). The library itself (settings, database,
# thumbnails; ~/.local/share/vision-tagger) is never deleted.
set -eu

data_home=${XDG_DATA_HOME:-$HOME/.local/share}
prefix=$data_home/mira
bin_dir=$HOME/.local/bin
action=do_install

usage() { sed -n '2,7s/^# \{0,1\}//p' "$0"; }
fail() { echo "install.sh: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case $1 in
    --prefix) [ $# -ge 2 ] || fail "--prefix needs a folder"; prefix=$2; shift 2 ;;
    --prefix=*) prefix=${1#--prefix=}; shift ;;
    --uninstall) action=do_uninstall; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "install.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -n "$prefix" ] || fail "--prefix needs a folder"
case $prefix in /*) ;; *) prefix=$(pwd)/$prefix ;; esac # it goes into the launchers

src=$(cd "$(dirname "$0")" && pwd)
app=$prefix/app
app_id=io.github.w1nte.mira
desktop_file=$data_home/applications/$app_id.desktop
icon_file=$data_home/icons/hicolor/512x512/apps/$app_id.png
marker="written by Mira's install.sh"

# 'it'\''s' quoting for a shell script
shell_quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# Quit a running Mira gracefully with the command line $1 (any version that knows quit-app).
quit_running() {
  echo "Closing Mira if it is running..."
  rc=0
  (cd "$HOME" && "$1" quit-app --timeout 120) || rc=$?
  [ "$rc" -eq 0 ] || fail "Mira is still running (exit code $rc). Close it, then run install.sh again."
}

write_launcher() { # name, program in the app folder
  target=$bin_dir/$1
  if [ -e "$target" ] && ! grep -q "$marker" "$target" 2>/dev/null; then
    echo "warning: $target exists and is not Mira's: left alone" >&2
    return 0
  fi
  printf '#!/bin/sh\n# %s (removed by install.sh --uninstall)\nexec %s "$@"\n' \
    "$marker" "$(shell_quote "$app/$2")" >"$target"
  chmod 755 "$target"
}

remove_launcher() {
  if [ -f "$bin_dir/$1" ] && grep -q "$marker" "$bin_dir/$1" 2>/dev/null; then
    rm -f "$bin_dir/$1"
  fi
}

refresh_menus() {
  if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q "$data_home/applications" 2>/dev/null || true
  fi
}

write_desktop_entry() {
  # Exec: quoted, with " ` $ \ escaped (and \ doubled once more for the string value)
  exec_path=$(printf '%s' "$app/Mira" | sed -e 's/[\\"`$]/\\\\&/g' -e 's/%/%%/g')
  mkdir -p "$(dirname "$desktop_file")" "$(dirname "$icon_file")"
  cp "$src/mira.png" "$icon_file"
  cat >"$desktop_file" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=Mira
GenericName=Photo library
Comment=Find, recognise and tag the people and things in your photos
Exec="$exec_path"
Icon=$icon_file
Terminal=false
Categories=Graphics;Photography;
Keywords=photos;pictures;faces;tags;search;
StartupNotify=true
EOF
  refresh_menus
}

do_uninstall() {
  if [ -x "$app/vision-tagger-cli" ]; then
    quit_running "$app/vision-tagger-cli"
  fi
  if [ -d "$app" ] || [ -f "$prefix/VERSION" ]; then
    rm -rf "$app" "$prefix/VERSION"
    rmdir "$prefix" 2>/dev/null || true
    echo "Removed Mira from $prefix"
  else
    echo "Mira is not installed in $prefix"
  fi
  remove_launcher mira
  remove_launcher vision-tagger-cli
  rm -f "$desktop_file" "$icon_file"
  refresh_menus
  echo "Your library (settings, database, thumbnails; by default $data_home/vision-tagger) is kept."
  echo "Delete that folder yourself to remove it as well."
}

do_install() {
  if [ ! -x "$src/app/Mira" ] || [ ! -x "$src/app/vision-tagger-cli" ]; then
    fail "there is no Mira app next to install.sh ($src/app)"
  fi
  if [ -d "$app" ] && [ "$(cd "$app" && pwd -P)" = "$(cd "$src/app" && pwd -P)" ]; then
    fail "run the install.sh of the downloaded archive, not the installed one"
  fi
  new_version=$(cat "$src/VERSION" 2>/dev/null || echo "?")
  if [ -d "$app" ]; then
    old_version=$(cat "$prefix/VERSION" 2>/dev/null || echo "?")
    echo "Updating Mira $old_version → $new_version in $prefix"
  else
    echo "Installing Mira $new_version into $prefix"
  fi

  # the new command line knows quit-app; a running Mira must not see its files replaced
  quit_running "$src/app/vision-tagger-cli"

  mkdir -p "$prefix"
  new=$prefix/.app-new-$$
  old=$prefix/.app-old-$$
  trap 'rm -rf "$new"' EXIT
  trap 'exit 130' INT TERM
  echo "Copying the files..."
  cp -R "$src/app" "$new"
  if [ -d "$app" ]; then
    # swap: a failed copy above leaves the old version untouched, and so does a failed swap
    mv "$app" "$old"
    if ! mv "$new" "$app"; then
      mv "$old" "$app"
      fail "could not replace $app; the old version is still installed"
    fi
    rm -rf "$old" || echo "warning: could not remove $old" >&2
  else
    mv "$new" "$app"
  fi
  printf '%s\n' "$new_version" >"$prefix/VERSION"

  mkdir -p "$bin_dir"
  write_launcher mira Mira
  write_launcher vision-tagger-cli vision-tagger-cli
  write_desktop_entry

  echo "Migrating the library's database (if there is one)..."
  rc=0
  (cd "$HOME" && "$app/vision-tagger-cli" migrate --desktop) || rc=$?
  case $rc in
    0) ;;
    3) echo "warning: the library's database is from a newer version of Mira, which this version cannot open." >&2 ;;
    *) echo "warning: the database could not be migrated (exit code $rc). Mira tries again when it starts; details are in logs/migrate.log in the library's data folder." >&2 ;;
  esac

  echo "Mira $new_version is installed: start it from the applications menu or with 'mira'."
  case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) echo "Note: $bin_dir is not in your PATH; start it with $bin_dir/mira" ;;
  esac
}

"$action"
