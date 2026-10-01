import { sourceLabel, sourceTone } from '../format.js'

export function SourceBadge({ source }) {
  if (!source || source === 'llm') return null
  return <span className={`badge ${sourceTone(source)}`}>{sourceLabel(source)}</span>
}
