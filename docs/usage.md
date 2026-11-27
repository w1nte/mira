# Using Mira

## Quick start

```powershell
uv run vision-tagger serve --open          # or: .\start.ps1
```

1. Open **Import** (sidebar, *System*). Add your photo folders with **Add a folder…** (a folder
   browser; you can also paste a path) and click **Scan now**. With *Analyse new photos right
   away* ticked, the pipeline runs afterwards: visual search, faces, ages, animals, properties
   and the search index.
2. Optionally put reference portraits into `references\<Person Name>\` and click
   **Import references** on the **Pipeline** page.
3. **Pipeline** shows the progress per stage, throughput, ETA and errors. There you can also
   choose the stages, a batch size and an optional limit, and **Pause**, **Resume** or **Stop**
   at any time.

Then name and clean up clusters under **Faces**, name your pets under **Animals**, explore
**Properties**, and filter or export in **Images**. **Home** shows what is in your library and your collections.

**Closing the app pauses a running pipeline** after the current image. The next start continues
it where it stopped, with the same stages and settings. Turn this off with
`pipeline.resume_on_start = false`.

Everything also works headless:

```powershell
uv run vision-tagger index "D:\Photos"
uv run vision-tagger run --stage faces --stage vlm --batch-size 16 --limit 500
uv run vision-tagger status
uv run vision-tagger models [pull]               # Ollama models: list / download missing ones
uv run vision-tagger pause | resume | stop       # from a second terminal
uv run vision-tagger refs                        # import reference DB + name clusters
uv run vision-tagger cluster [--full]
uv run vision-tagger pets [--full]               # group the detected animals again
uv run vision-tagger retry-errors --stage vlm
uv run vision-tagger requeue-outdated --stage vlm
uv run vision-tagger migrate [--check]           # database schema upgrade (with backup)
uv run vision-tagger export "D:\Export\Anna at the beach" --filter "p=1&t=location_type:beach" --layout date
uv run vision-tagger sidecars                    # XMP next to the originals (digiKam, Immich, …)
uv run vision-tagger search "Anna am Strand im Sommer, ohne Arbeitsfotos" --deep
uv run vision-tagger similar 123                 # photos that look like photo 123 (or a path)
uv run vision-tagger growup "Anna" anna-growing-up.webp  # grow-up animation
uv run vision-tagger db export | import FILE     # whole database (see "Moving a library")
```

`Ctrl+C` during `run` stops gracefully. Running `run` again continues where it stopped.

## The web UI

| Page | What you can do |
|---|---|
| **Home** | Your library at a glance: number of photos, people, places and years, *On this day* memories, recently added photos, and collections. Nudges you when face or animal groups still need a name, and while AI models are missing (with **Download now**). |
| **Collections** | A collection is a saved filter: it stays up to date as new photos arrive. Suggested ones (people, places, events, this year) appear by themselves; **Save as collection** in *Images* adds your own. Rename, reorder and delete them here. |
| **Import** | Local mode: your photo folders, with a folder browser, and **Scan now**; **Open folder** shows one in Explorer or Finder. Remote mode: upload photos, folders or ZIP files (see [Remote mode](remote.md)). Shows what is being indexed and analysed right now. |
| **Pipeline** | The technical part: start/pause/resume/stop runs (stages, batch size, limit), per-stage progress, ETA, errors (retry/skip), outdated results, jobs, recent runs, environment check (CUDA, Ollama, and the AI models with their state and **Download** buttons). |
| **Database** | Export the whole library database (backup, another computer) and import one (see [Moving a library](library.md#moving-a-library-to-another-computer-export--import)). **Missing files** lists photos that are not where Mira last saw them; scanning their new folder re-links them, **Remove** takes those that are gone for good out of the library. **Clear the library** starts over (two confirmations, a backup first). See [Cleaning up the library](library.md#cleaning-up-the-library). |
| **Log** | What Mira did, newest first: starts, scans, pipeline runs, and every warning and error with its details. Show only warnings and errors, **Download log**, **Open log file** in your text editor (on your own computer), **Report a bug**. See [The log and bug reports](#the-log-and-bug-reports). |
| **Faces** | All clusters with cover face and size. Name inline (autocomplete), accept reference suggestions, select several to **merge**. Views for unclustered and ignored faces. |
| **Cluster** | All faces of a cluster. Select faces to **split** into a new cluster, **assign to a person**, **move** to another cluster, or **remove**. **Auto-split** mixed clusters. Similar clusters with one-click merge. |
| **Animals** | Pet groups with picture and number of photos, by species. Name a group inline (typing another group's name merges the two), tick several to **merge**. Per group: all its photos to **move** to a pet (by name) or a new group, **take out** ("Not this one") or **remove**, similar groups with one-click merge, and the species (correct it when the detector took a dog for a bear). Views for ungrouped and removed animals. |
| **Persons** | Named persons. Rename (renaming to an existing name merges), merge, delete, manage references. Images with *all* / *any* of the selected persons. Per person: birth year (set or estimated), age today, a "through the years" face timeline, and photos with a doubtful date with suggested fixes (set the year, or keep the date and the age it gives). |
| **Properties** | Every extracted property grouped by topic, with counts. Tick values and show images with **all** / **any** of them (*any* keeps the other filters: "summer and (beach or mountains)"). Per property: sample images per value, **merge / rename / delete** values (remembered for future results). |
| **Images** | **Timeline slider** above the gallery: photos per month; click the first month, then the last, drag the band or its edges, or sweep across the bars (arrow keys move the handles). Gallery with infinite scroll and a filter builder: persons and pets (AND/OR, NOT), properties (AND/OR, NOT), date range, folder, camera, faces, text search, sort (with one person in the filter also by that person's age). Facets refine the current result; their **or** switch (or Ctrl+click / ⌘+click on a value) adds a value as an alternative, e.g. "beach or mountains" (see [Alternatives](search.md#alternatives-or)). A click on a photo shows it large over the gallery (**lightbox**): ← / → or a swipe for the previous / next photo (it loads more at the end), Esc, × or a click beside it to close (so does the browser's Back button), **Open details** for the photo's page; Ctrl+click (⌘+click) or a middle click opens the page in a new tab, and photos still covered as NSFW open their page. **Export** the result, with XMP metadata for other apps. **Select** to tick photos and mark them as NSFW or not NSFW, or give them all one date. Filters are part of the URL, so they can be bookmarked. |
| **Dates** | Photos that belong together, in groups, to correct wrong dates in one go (see [Correcting dates](features.md#correcting-dates)). *Need a look* lists the groups whose dates don't fit together or that only have file times, and photos with a camera's default date (e.g. `2000-12-31 22:00`). Tick photos, then **Use suggested dates**, **Set date** (a day, optionally a time) or **Shift dates** (the earliest photo gets the date, the others keep their distance). Filters from *Images* (folder, years, …) narrow it. |
| **Image** | Full image with face and animal boxes and names, all properties with source and confidence, the NSFW verdict (mark the photo as NSFW or not NSFW), metadata, duplicates, raw VLM JSON, per-stage status and re-run buttons. When a face doesn't fit the date: set the year, or keep the date and set the person's age (**Edit ages**). `←` / `→` open the previous / next photo of the current filter. **Show in folder** opens Explorer or Finder at the file (on your own computer). |
| **Quiz** | "Who's that?": name the largest unnamed clusters with one click (suggestions from references and look-alikes), or play the face quiz with score, streaks and "you vs. the AI"; wrong answers can fix misfiled faces. Keys 1–4 and Enter. |
| **Generations** | Every person with photos from different ages. Per person: a grow-up timelapse (play, scrub, download as animated WebP), "looks most like…" ranking, and a same-age comparison with any other person. |
| **Search** | Search box in the top bar on every page (`/` focuses it). Shows how the request was understood (persons, pets, required / preferred / excluded properties, alternatives such as "Anna or Ben", dates, keywords), ranks the results and, with **Deep search**, gives the LLM's reason for each. "Open in filter builder" turns the strict part into a gallery filter; results can be exported. |
| **Map** | Heatmap of the geotagged photos of the current filter (e.g. one person's photos), the places with their photo counts (click one for thumbnails, *Show photos* and renaming), **Photos in this area** (the map bounds as a gallery filter) and *Find places*. The view is kept in the URL (`#zoom/lat/lon`). |
| **Overview** | Timeline (year → month → day), breakdowns by folder, camera, file type, date reliability, work/personal, orientation, and a sortable metadata table. |

The sidebar groups the pages into *Library*, *People*, *Organize* and *System* and shows what
the pipeline is doing right now; at its bottom are *What's new* and **Report a bug**. On small
screens it folds away behind the menu button.

**Languages.** The UI speaks English, German, French, Spanish, Italian, Portuguese, Dutch,
Polish, Russian, Chinese and Japanese. It starts in your browser's language; the language menu at
the bottom of the sidebar changes it (kept in the `vt_lang` cookie). Property names and values
are translated too, while filters, exports and the database keep the English values, so links
and exported keywords stay the same in every language. Searches can be typed in any language.

**Light and dark mode.** The switch at the bottom of the sidebar picks the theme: *System*
(follows your operating system), *Light* or *Dark*. The choice is kept in a cookie (`vt_theme`),
so every page is rendered in the right theme from the start, without flashing.

**Install as an app.** Mira is a Progressive Web App. In Chrome or Edge, click
**Install app** in the sidebar (or the install icon in the address bar); on Android use *Add to
home screen*, on iOS *Share → Add to Home Screen*. It then opens in its own window, with its own
icon. Browsers allow this on `localhost` and over HTTPS, so a server needs HTTPS (see below).
Without a connection the app shows an offline page instead of a browser error.

## The log and bug reports

Mira writes down what it does in a log file: every start (with the version and the system), scans,
pipeline runs, jobs such as exports and imports, and above all every warning and error, with the
technical details (the traceback). When something goes wrong, the log shows what led up to it.

- **Where.** `logs/vision-tagger.log` in the data folder (the desktop app: see
  [where things are](installation.md#desktop-app-windows-macos-linux); Docker: `/data/logs` in
  the data volume). The desktop app, `vision-tagger serve` and the command line all write to the
  same file.
- **Size.** When the file reaches 5 MB, its content moves to `vision-tagger.log.1` (the older one
  to `.2`, and so on) and the oldest part is deleted, so the log never takes more than 20 MB. The
  size, the number of older files and the detail are settings (`[logging]`, see
  [Configuration](configuration.md)).
- **In the app.** *System → Log* shows the newest entries; *Warnings and errors* and *Errors* filter
  them, **Details** opens a traceback. **Download log** saves the whole log as one file. On your
  own computer, **Open log file** opens it in your text editor (Notepad, Console, …) and **Show in
  folder** shows it in Explorer or Finder. The *Pipeline* page links to the log next to its errors
  and failed jobs.

**Reporting a bug.** **Report a bug** (bottom of the sidebar, and on the Log page) opens a new
issue on GitHub with the bug report form, with your Mira version and system already filled in
(the operating system, Python, desktop app or Docker; no file names). Nothing is sent by itself:
you see the form in your browser and decide what to submit (a GitHub account is needed). Describe
what happened and add the downloaded log: drag the file into the *Log* box. The log contains the
names of files and folders in your library, so look through it first and remove what you don't
want to share.
