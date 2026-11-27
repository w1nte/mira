# Using the results in other applications

Exports carry what Mira found as **XMP**, the metadata standard photo applications
share. The originals are never modified.

| What | Where it ends up |
|---|---|
| Keywords: objects, place, event, season, …, pets and their species, and the people's names | `dc:subject` |
| Every property as a hierarchical keyword (`vision-tagger\|Location type\|beach`, `People\|Anna`) | `lr:hierarchicalSubject` |
| Caption of the vision model | `dc:description` |
| People, with the area of each face | `Iptc4xmpExt:PersonInImage`, MWG face regions (`mwg-rs:Regions`) |
| Capture date, including dates you corrected (not file-time guesses) | `exif:DateTimeOriginal`, `xmp:CreateDate`, `photoshop:DateCreated` |
| All raw values with confidence | `vt:Properties` (Mira namespace) |

Choose in the export dialog (or with `--xmp/--no-xmp`, `--embed/--no-embed`,
`--index/--no-index`):

- **XMP sidecar files** (`photo.jpg.xmp`) for digiKam, darktable, Immich, PhotoPrism, ExifTool,
  and Lightroom for raw files. `export.sidecar_naming = "replace"` writes `photo.xmp` instead.
- **Into the copies:** the XMP packet is added to exported JPEG and PNG copies without
  re-encoding the image. Lightroom, Apple Photos, Windows Explorer and Bridge read it. Hard and
  symbolic links are never changed (they *are* the originals), and an XMP packet that is already
  in a photo (e.g. a motion photo) is never replaced; the sidecar carries the data then.
- **Index files:** `vision-tagger.csv` (opens in Excel) and `vision-tagger.json` (for scripts)
  list every exported photo with its date, people, keywords, caption, face areas and all
  properties. Exporting again into the same folder updates them.

If your library is managed in place by digiKam, Immich or PhotoPrism, **Write XMP next to the
originals** (in the export dialog, or `vision-tagger sidecars`) puts a sidecar beside each
photo. A sidecar that another application wrote is never overwritten.

Checked with Exiv2, the metadata library of digiKam and darktable: keywords, hierarchy, caption,
people, date and named face regions are read back from sidecars and embedded copies.
