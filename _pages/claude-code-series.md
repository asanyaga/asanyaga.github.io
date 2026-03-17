---
title: "Building With Claude Code: A Developer's Guide to AI-Assisted Workflows"
layout: single
permalink: /tutorials/claude-code/
author_profile: true
excerpt: "A hands-on series that teaches you when to reach for CLAUDE.md vs. a skill vs. a hook vs. a subagent — and why."
---

This series teaches you how to use Claude Code for real software development — not as a feature walkthrough, but as a decision guide. We build a small app (Markd, a bookmarks API) along the way, and each post introduces a Claude Code capability at the moment you need it.

By the end, you'll know *when* to reach for CLAUDE.md vs. a skill vs. a hook vs. a subagent, and more importantly, *why*.

{% assign series_posts = site.posts | where: "series", "Building With Claude Code" | sort: "series_order" %}

{% for post in series_posts %}
### Part {{ post.series_order }}: [{{ post.title }}]({{ post.url }})
<small>{{ post.date | date: "%B %-d, %Y" }} &middot; {% assign words = post.content | number_of_words %}{{ words | divided_by: 200 }} min read</small>

{{ post.excerpt | strip_html | truncatewords: 30 }}

---
{% endfor %}
