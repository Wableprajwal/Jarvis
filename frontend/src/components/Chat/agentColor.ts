/** Map an agent name to its identity colour token. Stable per entity, never per rank. */
export function agentColorVar(agent: string | null | undefined): string {
  const a = (agent ?? '').toLowerCase();
  if (a.includes('orchestrat')) return 'var(--agent-orchestrator)';
  if (a.includes('research')) return 'var(--agent-research)';
  if (a.includes('code')) return 'var(--agent-code)';
  if (a.includes('ops')) return 'var(--agent-ops)';
  return 'var(--agent-default)';
}
