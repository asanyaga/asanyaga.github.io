---
title: "Perspectives"
layout: single
permalink: /perspectives/
author_profile: true
excerpt: "Opinion pieces on AI and the future of work."
---

{% assign posts = site.posts | where_exp: "post", "post.categories contains 'perspectives'" %}

{% if posts.size > 0 %}
{% for post in posts %}
### [{{ post.title }}]({{ post.url }})
<small>{{ post.date | date: "%B %-d, %Y" }} &middot; {% assign words = post.content | number_of_words %}{{ words | divided_by: 200 }} min read</small>

{{ post.excerpt | strip_html | truncatewords: 40 }}

---
{% endfor %}
{% else %}
Posts coming soon. In the meantime, check out the [tutorials](/tutorials/) or read more [about what I do](/about/).
{% endif %}
