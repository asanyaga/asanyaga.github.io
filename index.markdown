---
layout: splash
author_profile: true
header:
  overlay_color: "#1a1a2e"
excerpt: "Practical AI strategy and engineering for businesses that want to move beyond the hype."

feature_row:
  - title: "Tutorials"
    excerpt: "Hands-on technical guides. Currently a series on building AI agents from scratch in Python."
    url: /tutorials/
    btn_label: "Read Tutorials"
    btn_class: "btn--primary"
  - title: "AI Strategy"
    excerpt: "Guides for technical leaders navigating AI adoption. Separating hype from practical concerns."
    url: /ai-strategy/
    btn_label: "Read Guides"
    btn_class: "btn--primary"
  - title: "Perspectives"
    excerpt: "Opinion pieces on AI and the future of work. Critical thinking about where things are heading."
    url: /perspectives/
    btn_label: "Read Perspectives"
    btn_class: "btn--primary"
---

{% include feature_row %}

## Latest

{% assign latest = site.posts | first %}
{% if latest %}
### [{{ latest.title }}]({{ latest.url }})
<small>{{ latest.date | date: "%B %-d, %Y" }} &middot; {% assign words = latest.content | number_of_words %}{{ words | divided_by: 200 }} min read</small>

{{ latest.excerpt | strip_html | truncatewords: 40 }}
{% endif %}
