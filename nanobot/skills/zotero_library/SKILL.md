---
description: Zotero local-library workflow for finding papers, collections, and attachment PDFs
---

Use these tools when the user wants help with a Zotero library:

1. Use `list_zotero_collections` to inspect the library structure first if the user does not know the exact collection or topic organization.
2. Use `search_zotero_library` to search by title keyword, author, venue, or year.
3. Prefer passing the Zotero data directory (the folder containing `zotero.sqlite`).
4. If attachment paths are returned, use PDF-reading tools on the returned files for deeper reading and summarization.

Recommended workflow:
- list_zotero_collections → search_zotero_library → read_pdf / summary_pdf_file
