---
title: "Teaching Machines to Read"
date: 2026-06-21 08:00:00 +0300
categories: perspectives
tags: ai idp documents machine-learning
excerpt: "Ask someone to find the total on an invoice. They'll have it in seconds. Replicating that process in a machine involves five separate engineering problems, each one invisible until you try to solve it."
---

## What a Document Actually Is

Ask someone to open an invoice and find the total due. They'll have it in seconds. They don't think about how they did it. They glanced at the layout, skipped past the header, found the bold number near the bottom right, and read it. That process took less than three seconds. Nobody ever had to teach it to them.

Now try to teach it to a machine.

Intelligent Document Processing (IDP) is the discipline of automating what human readers do instinctively: reading a document, understanding what it is, and extracting meaning from it. IDP breaks into five stages: intake, parsing, classification, extraction, and indexing for retrieval. Each hides a layer of complexity that only becomes visible when you try to replicate it. Replicating that process, stage by stage, is what makes document automation so persistently difficult.

---

## Intake: Every Format Is a Different Problem

Every format a document arrives in is a completely different technical artifact. A native PDF stores instructions for where to render shapes on a page: not words, not sentences, not paragraphs as a computer would recognise them. A Word document has structural markup, but it varies by version, by author habits, and by whether styles were used consistently or ignored. A scanned document isn't a document at all from a machine's perspective. It's a photograph, with no text in it until an OCR engine runs and attempts to reconstruct what the camera captured, character by character, from pixels.

This is why production IDP systems don't have a single intake path. They route documents by format: native PDFs to one parser, scanned images to an OCR pipeline, Office files to another handler entirely, then reconcile the outputs into something consistent. The variation within each format means even that routing logic needs to account for exceptions. Get this wrong, and every stage downstream is working from incomplete or corrupted input, often without knowing it.

---

## Parsing: Humans See Structure. Machines See Coordinates.

When a person opens a document, they immediately perceive structure. Bold text means a heading or something important. A two-column layout is read left column first, then right. A table has rows and columns that relate to each other. A footnote belongs to the number it's anchored to, not to whatever text happens to be spatially adjacent. None of this is taught explicitly. It's absorbed from years of reading.

A machine gets none of this for free. A PDF doesn't store the concept of a heading. It stores a drawing instruction: render these glyphs, at these coordinates, in this font size. The parser has to infer that larger, bolder text probably signals a section start, but that inference fails when a designer has used large decorative text for aesthetic reasons, or when a heading is defined by position rather than weight.

Tables are worse. What a human reads as a grid with clear relationships is, inside a PDF, a set of lines drawn at specific coordinates and text placed nearby, with no semantic connection between them. When there are no grid lines at all and columns are implied only by whitespace alignment, the problem becomes pure visual inference. Tested against real-world documents, even the best purpose-built table extraction tools produce results that regularly surprise the people who built them.

The most significant recent advance here has been treating the page as an image rather than a text stream. Models like Microsoft's LayoutLM were trained jointly on text content, visual appearance, and the 2D position of every word on the page, learning to identify headings, tables, and figures the way a human reader would, rather than trying to reverse-engineer the file format. Vision-language models take this further, reasoning about layout from the rendered page directly. The trade-off is cost: processing every page visually is slower and more expensive, so most production systems use it selectively, reserving the heavier approach for the regions where layout inference is most likely to fail.

---

## Classification: An Expert Knows What They're Looking At

A finance director can tell at a glance whether a document is a supplier invoice or an internal credit note, and adjusts their expectations before reading a single line. That recognition is fast because they bring context the document never states explicitly.

A machine has to earn that same recognition from scratch, at scale, across every document type the business handles. The problem isn't distinguishing an invoice from a contract. It's the variation within a single category. An enterprise receives invoices from hundreds of suppliers. Some use the word "Invoice." Some say "Tax Invoice." Some say "Statement of Account" for what is effectively a bill. Same document type, dozens of layouts, no consistent signal a machine can anchor to.

For years, the standard answer was template libraries: a human would configure each document type, define what it looked like, and the system would match incoming documents against the library. It breaks the moment a supplier redesigns their letterhead, a new vendor joins, or a document arrives in a format the library doesn't cover. Modern approaches use language models that classify a document zero-shot, reasoning from content and structure rather than matching against a predefined template, and handle formats the system hasn't explicitly seen far better than template matching does. The failure modes shift rather than disappear, but the range of documents a system can handle without manual configuration expands significantly.

Misclassification propagates downstream silently. If the system reads a remittance advice as a purchase order, every downstream extraction runs against the wrong field map. The error is silent until a human notices a number doesn't reconcile, sometimes days later. Classification isn't a preprocessing step that can be treated as approximately correct. It's load-bearing infrastructure for everything that follows.

---

## Extraction: A Skilled Reader Knows Where to Look and What It Means

An accountant opening a balance sheet doesn't scan the entire page. Their eye goes directly to the right places: assets, then liabilities, then the equity section. They understand not just where numbers appear but what they mean: that a figure in the current liabilities row represents obligations due within twelve months, that it should be read against the current assets figure for the same period, and that a footnote three pages later might materially qualify the number they're looking at.

Teaching a machine to extract reliably is the stage where IDP most often fails in production. Rule-based systems, which mapped each field to a specific zone of a specific document type, were the backbone of first-generation IDP and still perform well for high-volume, low-variation types like standardised forms. They break when the variation exceeds what the rules anticipated, which in practice is constantly: a total that appears in a footer spanning multiple pages, a currency in a non-standard notation, two figures that both look like "total" (one pre-tax, one post) with no label to distinguish them.

LLM-based extraction handles this variation more gracefully, using language understanding to locate and interpret fields from any layout rather than a predefined coordinate map. Most production implementations pair this with per-field confidence scores, so that low-certainty extractions get flagged for human review rather than flowing downstream unchecked. The two approaches are increasingly used together: models handle the variation, rules provide guardrails on the outputs that matter most.

Extracting a value and understanding what it means are different problems. Pulling "Majid Al Futtaim Hypermarkets LLC" verbatim from a document is extraction. Knowing that this entity reconciles to "Carrefour" in a supplier ledger is normalisation: a downstream problem that requires domain knowledge, reference data, and a separate resolution step. Systems that try to solve both in a single pass tend to solve neither reliably.

---

## Chunking and Indexing: How Humans Navigate Long Documents

Nobody reads a ninety-page financial report from cover to cover looking for one specific figure. An analyst uses the table of contents to find the segment reporting section, skips to the relevant table, reads it in context, and cross-references the footnotes. They maintain a mental model of the document as they go. When they find a number, they know which section it came from, what it relates to, and where to look if it doesn't make sense.

Retrieval-Augmented Generation (RAG) attempts to replicate this: a query comes in, the system retrieves the most relevant passages from the index, and a model uses them to construct an answer. The quality of that answer depends entirely on how the documents were sliced and indexed in the first place.

The naive approach, cutting documents into fixed-size text blocks and indexing them, immediately discards the structural awareness that makes a human analyst effective. A chunk that begins mid-table has lost the column headers that give its numbers meaning. A chunk that ends halfway through a clause can't answer a question about that clause's implications. A financial statement sliced without reference to its own section structure will scatter related information across fragments that never appear together in retrieval.

Structure-aware chunking uses the document's own hierarchy (sections, headings, tables) as the natural unit of indexing rather than arbitrary text windows. For retrieval itself, combining keyword search with vector similarity (which captures conceptual relatedness rather than exact matches) outperforms either approach alone on real-world document collections. For long documents, a two-stage strategy mirrors how a human analyst would navigate the same report: a lightweight index over headings identifies the right region first, then fetches the full content of that section rather than pulling fragments from across the document and hoping the answer survives the assembly.

The indexing layer is where every compromise made earlier in the pipeline becomes visible. Parsing errors create broken context. Missed section boundaries produce chunks that mix unrelated content. When a retrieval failure surfaces as a hallucinated answer or a missed risk factor, the root cause is usually not the model. It's a decision made three stages earlier about how to represent the document.

---

## The Gap That AI Is Closing

For most of the history of document automation, the gap between human reading and machine reading was bridged with rules: templates that specified where to look for each field, trained on documents the system had seen before. This worked on narrow, predictable document types and broke on everything else.

What large language models bring to IDP is something closer to what humans actually do: reasoning from context, handling variation, inferring meaning from layout and language simultaneously rather than in rigid sequence. A model can read an unfamiliar invoice template and find the total due not because it was told where to look, but because it understands what invoices are and what totals look like.

That capability is real. But it doesn't make the pipeline disappear. A model fed malformed parser output still produces unreliable answers. Strong extraction still fails when classification was wrong. The stages are the same. What changes is how much variation each one can handle before it breaks.

Documents were built to communicate between people. Teaching machines to read them means reconstructing, step by step, the intuitions that readers apply without thinking.
