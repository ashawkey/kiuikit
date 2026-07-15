import { createElement } from 'react'
import type { ReactNode } from 'react'
import { createLowlight } from 'lowlight'
import type { Root, RootContent } from 'hast'

// Register only the grammars we map below (not highlight.js' full "common"
// set) to keep the bundle lean. Each import is a single language definition.
import bash from 'highlight.js/lib/languages/bash'
import c from 'highlight.js/lib/languages/c'
import cpp from 'highlight.js/lib/languages/cpp'
import csharp from 'highlight.js/lib/languages/csharp'
import css from 'highlight.js/lib/languages/css'
import dart from 'highlight.js/lib/languages/dart'
import go from 'highlight.js/lib/languages/go'
import ini from 'highlight.js/lib/languages/ini'
import java from 'highlight.js/lib/languages/java'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import kotlin from 'highlight.js/lib/languages/kotlin'
import less from 'highlight.js/lib/languages/less'
import lua from 'highlight.js/lib/languages/lua'
import markdown from 'highlight.js/lib/languages/markdown'
import php from 'highlight.js/lib/languages/php'
import python from 'highlight.js/lib/languages/python'
import r from 'highlight.js/lib/languages/r'
import ruby from 'highlight.js/lib/languages/ruby'
import rust from 'highlight.js/lib/languages/rust'
import scala from 'highlight.js/lib/languages/scala'
import scss from 'highlight.js/lib/languages/scss'
import sql from 'highlight.js/lib/languages/sql'
import swift from 'highlight.js/lib/languages/swift'
import typescript from 'highlight.js/lib/languages/typescript'
import xml from 'highlight.js/lib/languages/xml'
import yaml from 'highlight.js/lib/languages/yaml'

const lowlight = createLowlight({
  bash, c, cpp, csharp, css, dart, go, ini, java, javascript, json, kotlin,
  less, lua, markdown, php, python, r, ruby, rust, scala, scss, sql, swift,
  typescript, xml, yaml,
})

// File extension → highlight.js language id. Only entries whose language is in
// the common set matter; unknown extensions fall back to plain text.
const EXTENSION_LANGUAGE: Record<string, string> = {
  py: 'python',
  pyi: 'python',
  c: 'c',
  h: 'c',
  cc: 'cpp',
  cpp: 'cpp',
  cxx: 'cpp',
  hpp: 'cpp',
  hh: 'cpp',
  js: 'javascript',
  jsx: 'javascript',
  mjs: 'javascript',
  cjs: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  json: 'json',
  css: 'css',
  scss: 'scss',
  less: 'less',
  html: 'xml',
  xml: 'xml',
  md: 'markdown',
  markdown: 'markdown',
  sh: 'bash',
  bash: 'bash',
  zsh: 'bash',
  yml: 'yaml',
  yaml: 'yaml',
  toml: 'ini',
  ini: 'ini',
  go: 'go',
  rs: 'rust',
  java: 'java',
  kt: 'kotlin',
  swift: 'swift',
  rb: 'ruby',
  php: 'php',
  cs: 'csharp',
  sql: 'sql',
  lua: 'lua',
  r: 'r',
  scala: 'scala',
  dart: 'dart',
}

// Resolve a highlight.js language from a file path, or null when the extension
// is unknown / unsupported so callers can render plain text.
export function languageForPath(path: string | undefined): string | null {
  if (!path) return null
  const name = path.split(/[\\/]/).pop() ?? path
  const dot = name.lastIndexOf('.')
  if (dot <= 0) return null
  const ext = name.slice(dot + 1).toLowerCase()
  const lang = EXTENSION_LANGUAGE[ext]
  return lang && lowlight.registered(lang) ? lang : null
}

let keyCounter = 0

function hastToReact(nodes: RootContent[]): ReactNode[] {
  return nodes.map((node) => {
    if (node.type === 'text') return node.value
    if (node.type === 'element') {
      const className = node.properties?.className
      const cls = Array.isArray(className) ? className.join(' ') : undefined
      return createElement(
        node.tagName,
        { key: keyCounter++, className: cls },
        ...hastToReact(node.children as RootContent[]),
      )
    }
    return null
  })
}

// Highlight one source line for the given language, returning token spans
// (highlight.js `hljs-*` classes). Falls back to the raw string on failure so
// a bad grammar match never blanks the diff.
export function highlightLine(line: string, language: string | null): ReactNode {
  if (!language) return line
  try {
    const tree = lowlight.highlight(language, line) as Root
    return hastToReact(tree.children as RootContent[])
  } catch {
    return line
  }
}
