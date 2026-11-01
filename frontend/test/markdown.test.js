import { describe, expect, it } from 'vitest'
import { escapeHtml, renderMarkdown } from '../src/markdown.js'

describe('escapeHtml', () => {
  it('escapes every HTML metacharacter', () => {
    expect(escapeHtml('<b>&"\'</b>')).toBe(
      '&lt;b&gt;&amp;&quot;&#39;&lt;/b&gt;',
    )
  })
})

describe('renderMarkdown', () => {
  it('wraps text in paragraphs and joins lines with <br/>', () => {
    expect(renderMarkdown('hello\nworld')).toBe('<p>hello<br/>world</p>')
    expect(renderMarkdown('first\n\nsecond')).toBe('<p>first</p><p>second</p>')
  })

  it('renders bold, italic and inline code', () => {
    expect(renderMarkdown('a **bold** word')).toBe('<p>a <strong>bold</strong> word</p>')
    expect(renderMarkdown('an *italic* word')).toBe('<p>an <em>italic</em> word</p>')
    expect(renderMarkdown('use `_x` here')).toContain('<code>_x</code>')
  })

  it('renders bullet and numbered lists', () => {
    expect(renderMarkdown('- one\n- two')).toBe('<ul><li>one</li><li>two</li></ul>')
    expect(renderMarkdown('1. one\n2. two')).toBe('<ol><li>one</li><li>two</li></ol>')
  })

  it('links http(s) and mailto targets only', () => {
    const good = renderMarkdown('[site](https://example.com)')
    expect(good).toContain('<a href="https://example.com"')
    expect(renderMarkdown('[mail](mailto:a@b.c)')).toContain('mailto:a@b.c')
    const bad = renderMarkdown('[x](javascript:alert(1))')
    expect(bad).not.toContain('<a ')
    expect(bad).toContain('[x](javascript:alert(1))')
  })

  it('never lets markup escape as HTML (XSS-safe)', () => {
    const out = renderMarkdown('<script>alert("x")</script> **hi**')
    expect(out).not.toContain('<script>')
    expect(out).toContain('&lt;script&gt;')
    expect(out).toContain('<strong>hi</strong>')
    const image = renderMarkdown('<img src=x onerror=alert(1)>')
    expect(image).not.toContain('<img')
  })

  it('handles empty input', () => {
    expect(renderMarkdown('')).toBe('')
    expect(renderMarkdown('   \n  ')).toBe('')
    expect(renderMarkdown(null)).toBe('')
  })
})
