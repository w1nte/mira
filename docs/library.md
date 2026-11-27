# Your library

## Moving a library to another computer (export / import)

Everything Mira knows about your photos (index, faces, clusters, names, properties, collections,
settings) is in one database file. The **Database** page (sidebar, *System*) exports and imports
it; the photos themselves are not part of it, copy them separately.

- **Export** downloads `vision-tagger-YYYY-MM-DD.sqlite3`: a consistent snapshot, made while
  Mira keeps running, also while the pipeline works (`VACUUM INTO`, so it is compact and one
  self-contained file).
- **Import** replaces the current library with such a file.
  - It is refused while the pipeline runs or is paused, or while another job runs: stop the
    pipeline first.
  - The file is checked first (SQLite, not damaged, a Mira database, a schema this version
    knows). Files from older versions are migrated, files from newer versions are refused.
  - A backup of the current database stays next to it: `vision_tagger.sqlite3.bak-import-<time>`.
  - Thumbnails and face crops are cleared (image ids now mean other photos) and made again on
    demand. The pipeline state of the other computer (a run that was active there) is reset.
  - The library folders become the imported folders that exist on this computer plus the ones
    configured before; remote mode keeps its upload folder. Imported folders on a network share
    (`\\server\share`) none of your folders is on are not added (the import message lists them):
    add them on the **Import** page if you trust them, then scan.
- **Missing files.** After an import Mira checks where every photo is, not only those below
  the current folders. Photos that are not where the other computer had them are flagged as
  missing and listed under **Database → Missing files** (with their faces and names), with a
  **Check again** button. The gallery, search and statistics leave missing photos out, as they
  do for files that vanished; their faces stay in *Faces* and *Persons*.
- **Re-linking.** Add or scan the folder where the photos are now (in remote mode: upload them
  again). A new file whose perceptual hash matches a missing photo (same pHash, dHash within
  `index.dhash_max_distance`) is linked to that photo instead of being added again: same id,
  all faces, names, properties and manual dates kept. The scan result counts these as *moved*.

From the command line (close the web UI first for an import):

```powershell
uv run vision-tagger db export D:\Backup                 # writes vision-tagger-<date>.sqlite3
uv run vision-tagger db import D:\Backup\vision-tagger-2026-11-26.sqlite3
uv run vision-tagger index "E:\Photos"                  # re-links the photos found there
```

## Cleaning up the library

The **Database** page also cleans up. Like an import, removing photos and clearing the library
are refused while the pipeline runs or is paused, or while another job runs: stop the pipeline
first. Neither ever deletes, moves or changes a photo file.

- **Removing missing photos.** Photos that are gone for good (deleted, or on a drive you no
  longer have) can be removed under **Database → Missing files**: **Remove** next to a photo,
  **Remove all missing photos (N)** for the whole list, or **Remove from library** on the page of
  a missing photo. Each asks first.
  - Only photos flagged as missing can be removed. Right before deleting, Mira looks for each
    file once more: a photo whose file is there again (a drive was connected meanwhile) is
    skipped and counted, never removed. So is a photo with a duplicate file that still exists;
    **Check again** makes that copy its file.
  - Removed with a photo: its faces, animals, properties, search index entries, duplicate
    records, pipeline status and cached thumbnails and crops.
  - Cleaned up: face clusters and unnamed animal groups without photos are deleted, the others
    get new counts and covers; places without photos are deleted unless you named them.
  - **Named persons and named pets are never deleted.** They keep their names and reference
    faces and just have fewer photos.
  - If a removed file turns up again later, a scan adds it as a new photo, analysed from the
    start.
- **Clearing the library** starts over with an empty library: **Database → Danger zone → Clear
  the library…**.
  - First step: a page lists what is deleted, with the current numbers: all photos' data, faces,
    clusters, persons and their names, animals and pets, properties, collections, places, the
    search index and the pipeline history. It also lists what is kept. Tick *I understand* to
    continue.
  - Second step: type the number of photos in the library to enable **Clear the library now**.
    A number works in every language, and it must still match when you click: if photos were
    added meanwhile, Mira refuses. The server checks both steps again.
  - Kept: the photo files, your library folders (**Scan now** on the Import page builds the
    library again), `config.toml`, the language and the theme. In remote mode the uploaded files
    stay in the upload folder and come back with the next scan of it (after the next upload).
  - First, a backup of the database is written next to it:
    `vision_tagger.sqlite3.bak-clear-<time>`. Importing it on the Database page brings the
    library back. Thumbnails and crops are cleared, and browsers drop their cached copies.
  - From the command line: `vision-tagger db clear` asks the same two questions (a yes, then the
    number of photos); `--yes --yes` (`-yy`) answers both, for scripts. Close the web UI first.
- **Show in Explorer / Finder.** When Mira runs on the computer in front of you (the desktop app,
  or `vision-tagger serve` opened on the same computer), buttons open the file manager:
  - **Show in folder** on a photo's page selects the file in Explorer (Windows) or Finder
    (macOS). On Linux the file manager opens its folder. For a missing photo the button is off.
  - **Open folder** next to each library folder on the Import page.
  - **Show database file** and **Open data folder** on the Database page.
  - The buttons appear only in local mode and for a browser on that same computer: a request
    from `127.0.0.1` or `::1`, under a local host name, not through a reverse proxy. On Linux
    there must be a graphical session. They never appear in remote mode, in Docker, or from
    another device on your network. The browser only says *what* to show (a photo, a library
    folder, the database), never a path.

## Data

Everything lives in `data/`:

- `vision_tagger.sqlite3`: index, job state, faces, clusters, persons, animals, pets,
  properties.
- `cache/thumbs`, `cache/faces`, `cache/animals`: thumbnails, face and animal crops. These can be deleted at any time.
  - Gallery thumbnails are written while indexing, from the image that is decoded for duplicate
    detection anyway (about 5 % extra indexing time).
  - **Scan & index** (or `vision-tagger index`) also creates the thumbnails that are missing,
    e.g. for photos indexed with an older version or after deleting the cache.
  - HEIC photos use their embedded preview when it is large enough (a few milliseconds instead
    of decoding 12 megapixels).
  - Turn this off with `index.thumbnails = false`.
- `models/geo`: the converted GeoNames towns for place names (downloaded once).

- `uploads/`: unfinished uploads (remote mode); finished ones are moved into the upload folder.
- `transfer/`: temporary files of database exports and imports (removed afterwards).
- `vision_tagger.sqlite3.bak-*`: backups made before a migration, an import or clearing the
  library. Delete old ones when you no longer need them.
- `secret.key`: the key that signs login cookies. Delete it to log everyone out.
- `update-check.json`: the last update check and the version you dismissed on Home.

Main tables: `images` (+ `metadata` JSON), `duplicate_files`, `stage_status`, `runs`, `faces`,
`clusters`, `persons`, `person_references`, `image_properties`, `property_aliases`,
`vlm_results`, `collections`, `places`, `geocode_cache`, `animals`, `pets`.
