---
title: "Guides"
layout: single
permalink: /tutorials/
classes: wide
excerpt: "Hands-on guides based on things I'm building and learning."
---

## Series

- **[Building With Claude Code](/tutorials/claude-code/)** — A hands-on series on using Claude Code for real software development. Learn when to reach for CLAUDE.md, skills, hooks, and subagents.
- **[From Prompts to Agents](/tutorials/ai-agents/)** — Build an AI agent from scratch in Python, step by step.

---

## All Posts

{% assign tutorials = site.posts | where_exp: "post", "post.categories contains 'tutorials'" %}
{% for post in tutorials %}
### [{{ post.title }}]({{ post.url }})
<small>{{ post.date | date: "%B %-d, %Y" }} &middot; {% assign words = post.content | number_of_words %}{{ words | divided_by: 200 }} min read</small>

{{ post.excerpt | strip_html | truncatewords: 40 }}

---
{% endfor %}
