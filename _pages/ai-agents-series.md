---
title: "From Prompts to Agents: Building AI Agents from Scratch"
layout: single
permalink: /tutorials/ai-agents/
author_profile: true
excerpt: "A 6-part tutorial series building an AI agent step by step in Python."
---

This series walks through building an AI agent from the ground up. We start with a simple prompt-response loop and progressively add tools, memory, reasoning, planning, and observability.

Each tutorial builds on the previous one, using a code review assistant as our running example.

{% assign series_posts = site.posts | where: "series", "From Prompts to Agents" | sort: "series_order" %}

{% for post in series_posts %}
### Part {{ post.series_order }}: [{{ post.title }}]({{ post.url }})
<small>{{ post.date | date: "%B %-d, %Y" }} &middot; {% assign words = post.content | number_of_words %}{{ words | divided_by: 200 }} min read</small>

{{ post.excerpt | strip_html | truncatewords: 30 }}

---
{% endfor %}
