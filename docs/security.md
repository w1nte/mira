# Security

In local mode the UI is meant for your own machine. It listens on `127.0.0.1`, only answers
requests whose `Host` is this machine (blocks DNS rebinding) and rejects form submissions coming
from other websites (CSRF). If you expose it on your network (`web.host = "0.0.0.0"`), add the
machine's name to `web.allowed_hosts` and set a password: `vision-tagger serve` refuses to listen
on such an address, or to run in remote mode, without one. The `Host` check only stops browsers;
it is no protection against other computers on the network.

Mira only ever opens image files as photos, judged by the file's own name, and in remote mode
only files below the upload folder. An imported database can't make it show other files.

The buttons that open Explorer or Finder only appear for, and only answer, a browser on the
server's own computer (local mode, a loopback address, a local host name, no proxy headers). The
request names a photo or folder by id; the server looks the path up itself.

In remote mode any host name is accepted (it runs behind your own domain or proxy), while
cross-site form submissions are still rejected. Set a password and use HTTPS (see
[Password and HTTPS](remote.md#password-and-https)). Login cookies are `HttpOnly`, `SameSite=Lax` and
signed with a key stored in the data folder.
