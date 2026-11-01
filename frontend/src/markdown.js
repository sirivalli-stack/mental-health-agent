/**
 * Safe mini-markdown renderer - pure, dependency-free, unit-testable.
 *
 * Every piece of input is HTML-escaped *before* any formatting is applied,
 * so the output is safe to inject via dangerouslySetInnerHTML. Links are
 * only produced for http(s) and mailto targets; anything else renders as
 * plain text (no javascript: URLs, no event handlers).
 *
 * Supported: paragraphs, line breaks, **bold**, *italic* / _italic_,
 * `inline code`, - / * bullet lists, 1. numbered lists, [text](url).
 */

export function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function safeUrl(url) {
  const trimmed = String(url ?? '').trim()
  if (/^(https?:\/\/|mailto:)/i.test(trimmed)) return trimmed
  return null
}

function inline(escaped) {
  let out = escaped
  // inline code first so its contents are not re-formatted
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>')
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  out = out.replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, '$1<em>$2</em>')
  out = out.replace(/(^|[\s(])_([^_\n]+)_(?![\w])/g, '$1<em>$2</em>')
  out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, label, url) => {
    const href = safeUrl(url)
    return href
      ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${label}</a>`
      : match
  })
  return out
}

export function renderMarkdown(text) {
  const source = String(text ?? '')
  if (!source.trim()) return ''

  const lines = source.split(/\r?\n/)
  const blocks = []
  let list = null // { type: 'ul' | 'ol', items: [] }
  let paragraph = []

  const flushParagraph = () => {
    if (paragraph.length) {
      blocks.push(`<p>${inline(escapeHtml(paragraph.join('\n'))).replace(/\n/g, '<br/>')}</p>`)
      paragraph = []
    }
  }
  const flushList = () => {
    if (list) {
      const tag = list.type
      blocks.push(`<${tag}>${list.items.map((i) => `<li>${i}</li>`).join('')}</${tag}>`)
      list = null
    }
  }

  for (const line of lines) {
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line)
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line)
    if (bullet || numbered) {
      flushParagraph()
      const type = bullet ? 'ul' : 'ol'
      if (!list || list.type !== type) {
        flushList()
        list = { type, items: [] }
      }
      list.items.push(inline(escapeHtml((bullet || numbered)[1])))
      continue
    }
    if (!line.trim()) {
      flushList()
      flushParagraph()
      continue
    }
    flushList()
    paragraph.push(line)
  }
  flushList()
  flushParagraph()
  return blocks.join('')
}
