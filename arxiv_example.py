import arxiv

# Construct the default API client.
client = arxiv.Client()

# Search for the 10 most recent articles matching the keyword "quantum."
search = arxiv.Search(
  query = "ti:(reasoning AND llm)",
  max_results = 3,
  sort_by = arxiv.SortCriterion.Relevance,
  sort_order=arxiv.SortOrder.Descending
)

results = client.results(search)

# `results` is a generator; you can iterate over its elements one by one...
for r in results:
  print(r.title,":\n",r.summary,"\n")
  print(r.pdf_url)
#   r.download_pdf(dirpath="./newbies", filename=r.title+".pdf")