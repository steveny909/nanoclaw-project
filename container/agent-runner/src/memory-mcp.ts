/**
 * Stdio MCP Server for PostgreSQL + pgvector memory layer.
 * Provides semantic memory tools to NanoClaw container agents.
 * Reads connection info from environment variables.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import pg from 'pg';

const { Pool } = pg;

const POSTGRES_URL = process.env.NANOCLAW_POSTGRES_URL!;
const INSTANCE_ID = process.env.NANOCLAW_INSTANCE_ID!;
const GROUP_ID = process.env.NANOCLAW_GROUP_ID!;
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'host.docker.internal:11434';
const EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL || 'nomic-embed-text';

const pool = new Pool({ connectionString: POSTGRES_URL });

async function getEmbedding(text: string): Promise<number[]> {
  const res = await fetch(`http://${OLLAMA_HOST}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: text }),
  });
  if (!res.ok) throw new Error(`Embedding request failed: ${res.status}`);
  const data = (await res.json()) as { embedding: number[] };
  return data.embedding;
}

function vectorLiteral(v: number[]): string {
  return `[${v.join(',')}]`;
}

const server = new McpServer({
  name: 'memory',
  version: '1.0.0',
});

// Tool 1: memory_add — store a core fact with key, value, and embedding
server.tool(
  'memory_add',
  'Store or update a core fact in long-term memory. Use this to remember important information, decisions, user preferences, and context that should persist across sessions.',
  {
    key: z.string().describe('A short descriptive key for this memory (e.g. "user_preference_language")'),
    value: z.string().describe('The fact or information to store'),
  },
  async (args) => {
    try {
      const embedding = await getEmbedding(args.value);
      await pool.query(
        `INSERT INTO core_memories (group_id, instance_id, key, value, embedding, updated_at)
         VALUES ($1, $2, $3, $4, $5::vector, NOW())
         ON CONFLICT (group_id, instance_id, key)
         DO UPDATE SET value = $4, embedding = $5::vector, updated_at = NOW()`,
        [GROUP_ID, INSTANCE_ID, args.key, args.value, vectorLiteral(embedding)],
      );
      return { content: [{ type: 'text' as const, text: `Stored memory: ${args.key}` }] };
    } catch (e) {
      return { content: [{ type: 'text' as const, text: `Error storing memory: ${e}` }], isError: true };
    }
  },
);

// Tool 2: memory_search — semantic search across core_memories
server.tool(
  'memory_search',
  'Search long-term memory for relevant facts using semantic similarity. Use this before starting tasks to recall context, preferences, and past decisions.',
  {
    query: z.string().describe('What to search for in memory'),
    limit: z.number().optional().default(5).describe('Max results to return (default 5)'),
  },
  async (args) => {
    try {
      const embedding = await getEmbedding(args.query);
      const result = await pool.query(
        `SELECT key, value, 1 - (embedding <=> $1::vector) AS similarity
         FROM core_memories
         WHERE group_id = $2 AND instance_id = $3 AND embedding IS NOT NULL
         ORDER BY embedding <=> $1::vector
         LIMIT $4`,
        [vectorLiteral(embedding), GROUP_ID, INSTANCE_ID, args.limit ?? 5],
      );
      if (result.rows.length === 0) {
        return { content: [{ type: 'text' as const, text: 'No memories found.' }] };
      }
      const formatted = result.rows
        .map((r: { key: string; value: string; similarity: number }) =>
          `[${(r.similarity * 100).toFixed(1)}%] ${r.key}: ${r.value}`)
        .join('\n');
      return { content: [{ type: 'text' as const, text: formatted }] };
    } catch (e) {
      return { content: [{ type: 'text' as const, text: `Error searching memory: ${e}` }], isError: true };
    }
  },
);

// Tool 3: conversation_embed — embed and store a conversation turn
server.tool(
  'conversation_embed',
  'Store an important conversation turn with its embedding for later semantic recall. Use this for significant exchanges worth remembering.',
  {
    session_id: z.string().describe('Current session identifier'),
    role: z.enum(['user', 'assistant']).describe('Who said this'),
    content: z.string().describe('The message content to store'),
  },
  async (args) => {
    try {
      const embedding = await getEmbedding(args.content);
      await pool.query(
        `INSERT INTO conversation_memory (group_id, instance_id, session_id, role, content, embedding)
         VALUES ($1, $2, $3, $4, $5, $6::vector)`,
        [GROUP_ID, INSTANCE_ID, args.session_id, args.role, args.content, vectorLiteral(embedding)],
      );
      return { content: [{ type: 'text' as const, text: `Stored ${args.role} message in conversation memory.` }] };
    } catch (e) {
      return { content: [{ type: 'text' as const, text: `Error storing conversation: ${e}` }], isError: true };
    }
  },
);

// Tool 4: conversation_recall — retrieve relevant past conversation turns
server.tool(
  'conversation_recall',
  'Retrieve past conversation turns that are semantically relevant to a query. Use this to recall what was discussed previously.',
  {
    query: z.string().describe('What to search for in past conversations'),
    limit: z.number().optional().default(10).describe('Max results to return (default 10)'),
  },
  async (args) => {
    try {
      const embedding = await getEmbedding(args.query);
      const result = await pool.query(
        `SELECT session_id, role, content, created_at,
                1 - (embedding <=> $1::vector) AS similarity
         FROM conversation_memory
         WHERE group_id = $2 AND instance_id = $3 AND embedding IS NOT NULL
         ORDER BY embedding <=> $1::vector
         LIMIT $4`,
        [vectorLiteral(embedding), GROUP_ID, INSTANCE_ID, args.limit ?? 10],
      );
      if (result.rows.length === 0) {
        return { content: [{ type: 'text' as const, text: 'No past conversations found.' }] };
      }
      const formatted = result.rows
        .map((r: { similarity: number; role: string; content: string; session_id: string; created_at: string }) =>
          `[${(r.similarity * 100).toFixed(1)}%] [${r.session_id}] ${r.role}: ${r.content.slice(0, 200)}`)
        .join('\n');
      return { content: [{ type: 'text' as const, text: formatted }] };
    } catch (e) {
      return { content: [{ type: 'text' as const, text: `Error recalling conversations: ${e}` }], isError: true };
    }
  },
);

// Tool 5: session_summarize — write a session summary with embedding
server.tool(
  'session_summarize',
  'Create a summary of the current session for future reference. Use this when a session is winding down to preserve key context.',
  {
    session_id: z.string().describe('Session identifier to summarize'),
    summary: z.string().describe('A concise summary of what happened in this session'),
    key_facts: z.array(z.string()).optional().describe('Array of key facts or decisions from this session'),
  },
  async (args) => {
    try {
      const embedding = await getEmbedding(args.summary);
      const keyFacts = JSON.stringify(args.key_facts ?? []);
      await pool.query(
        `INSERT INTO session_summaries (group_id, instance_id, session_id, summary, key_facts, embedding, ended_at)
         VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, NOW())
         ON CONFLICT (session_id)
         DO UPDATE SET summary = $4, key_facts = $5::jsonb, embedding = $6::vector, ended_at = NOW()`,
        [GROUP_ID, INSTANCE_ID, args.session_id, args.summary, keyFacts, vectorLiteral(embedding)],
      );
      return { content: [{ type: 'text' as const, text: `Session ${args.session_id} summary saved.` }] };
    } catch (e) {
      return { content: [{ type: 'text' as const, text: `Error saving session summary: ${e}` }], isError: true };
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
