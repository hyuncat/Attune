const downloadUrl = import.meta.env.VITE_MAC_DOWNLOAD_URL?.trim();

if (downloadUrl) {
  for (const link of document.querySelectorAll("a[data-mac-download]")) {
    link.href = downloadUrl;
  }
}
