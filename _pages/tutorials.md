---
title: "Tutorials"
layout: collection
permalink: /tutorials/
collection: posts
entries_layout: list
classes: wide
taxonomy: tutorials
excerpt: "Hands-on technical tutorials on building AI applications."
---

{% assign tutorials = site.posts | where_exp: "post", "post.categories contains 'tutorials'" %}
{% for post in tutorials %}
### [{{ post.title }}]({{ post.url }})
<small>{{ post.date | date: "%B %-d, %Y" }} &middot; {% assign words = post.content | number_of_words %}{{ words | divided_by: 200 }} min read</small>

{{ post.excerpt | strip_html | truncatewords: 40 }}

---
{% endfor %}
