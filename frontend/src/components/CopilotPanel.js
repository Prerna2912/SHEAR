import { useState, useRef, useEffect, useCallback } from 'react';
import { getResponse, QUICK_QUESTIONS, buildContext } from './CopilotEngine';

function TypingDots() {
  return (
    <div className="flex gap-1 items-center py-1 px-1">
      {[0, 1, 2].map(i => (
        <div
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-teal-400"
          style={{ animation: `bounce 1.2s ${i * 0.2}s ease-in-out infinite` }}
        />
      ))}
      <style>{`@keyframes bounce{0%,80%,100%{transform:translateY(0)}40%{transform:translateY(-5px)}}`}</style>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-up`}>
      {!isUser && (
        <div className="w-6 h-6 rounded-full bg-gradient-to-br from-teal-500 to-indigo-500 flex items-center justify-center text-white text-xs font-bold shrink-0 mr-2 mt-1">S</div>
      )}
      <div
        className={`max-w-[85%] px-3 py-2.5 text-xs leading-relaxed ${isUser ? 'chat-user text-teal-100' : 'chat-ai text-slate-200'}`}
        dangerouslySetInnerHTML={{
          __html: msg.content
            .replace(/\*\*(.*?)\*\*/g, '<strong class="text-white">$1</strong>')
            .replace(/`(.*?)`/g, '<code class="bg-white/10 px-1 rounded text-teal-300 font-mono text-[11px]">$1</code>')
            .replace(/\n/g, '<br/>')
            .replace(/•/g, '·'),
        }}
      />
    </div>
  );
}

export default function CopilotPanel({ result, cf4, explorerResult, af1, geometryType, params, isOpen, onClose }) {
  const [messages,  setMessages]  = useState([{
    role: 'assistant',
    content: 'Hello! I\'m SHEAR Copilot.\n\nRun a V1 analysis and I\'ll help you interpret the stress field, uncertainty, backscatter, and parameter sensitivity in plain engineering language.\n\nOr ask me anything about the methodology.',
  }]);
  const [input,     setInput]     = useState('');
  const [typing,    setTyping]    = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);

  const context = result
    ? buildContext({ geometryType, params, result, cf4, explorerResult, af1 })
    : null;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, typing]);

  useEffect(() => {
    if (isOpen) setTimeout(() => inputRef.current?.focus(), 100);
  }, [isOpen]);

  // Auto-greet when a new result arrives
  useEffect(() => {
    if (!result || !context) return;
    const greeting = `Analysis complete for **${geometryType}**.\n\nPeak τ = **${context.peakTau.toFixed(4)}** · Backscatter = **${context.backscatterFrac != null ? (context.backscatterFrac*100).toFixed(1)+'%' : '—'}** · Regime = **${context.dominantRegime ?? '—'}**${context.isOOD ? '\n\n⚠️ **OOD warning** — geometry is outside training distribution.' : ''}\n\nAsk me anything, or pick a quick question below.`;
    setMessages(prev => {
      if (prev.some(m => m._auto && m._geomType === geometryType)) return prev;
      return [...prev, { role: 'assistant', content: greeting, _auto: true, _geomType: geometryType }];
    });
  }, [result]); // eslint-disable-line react-hooks/exhaustive-deps

  const send = useCallback(async (question) => {
    const q = question.trim();
    if (!q) return;
    setInput('');
    setMessages(prev => {
      const updated = [...prev, { role: 'user', content: q }];
      // Fire async separately so we have the updated history
      (async () => {
        setTyping(true);
        try {
          const answer = await getResponse(q, context, updated);
          setMessages(m => [...m, { role: 'assistant', content: answer }]);
        } catch (e) {
          setMessages(m => [...m, { role: 'assistant', content: `Error: ${e.message}` }]);
        } finally {
          setTyping(false);
        }
      })();
      return updated;
    });
  }, [context]);

  if (!isOpen) return null;

  return (
    <div className="fixed right-0 top-0 h-screen w-80 z-50 flex flex-col border-l border-white/[0.07] shadow-2xl animate-fade-up" style={{ background: 'rgba(4,6,12,0.82)', backdropFilter: 'blur(32px) saturate(160%)', WebkitBackdropFilter: 'blur(32px) saturate(160%)' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3.5 border-b border-white/[0.06]">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-teal-400 to-indigo-500 flex items-center justify-center text-white text-xs font-bold">S</div>
          <div>
            <div className="text-sm font-semibold font-display text-white">SHEAR Copilot</div>
            <div className="text-xs text-slate-500">{process.env.REACT_APP_CLAUDE_API_KEY ? 'Powered by Claude' : 'Engineering AI'}</div>
          </div>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-300 transition-colors text-lg">✕</button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-3 py-3 flex flex-col gap-3">
        {messages.map((m, i) => <MessageBubble key={i} msg={m} />)}
        {typing && (
          <div className="flex justify-start">
            <div className="w-6 h-6 rounded-full bg-gradient-to-br from-teal-500 to-indigo-500 flex items-center justify-center text-white text-xs font-bold shrink-0 mr-2 mt-1">S</div>
            <div className="chat-ai px-3 py-2">
              <TypingDots />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Quick questions */}
      <div className="px-3 py-2 border-t border-white/[0.04]">
        <div className="text-xs text-slate-600 mb-1.5">Quick questions</div>
        <div className="flex flex-wrap gap-1">
          {QUICK_QUESTIONS.slice(0, 4).map(q => (
            <button
              key={q}
              onClick={() => send(q)}
              disabled={typing}
              className="px-2 py-1 rounded-full text-xs bg-white/[0.05] border border-white/[0.07] text-slate-400 hover:text-teal-300 hover:border-teal-500/40 transition-all disabled:opacity-40"
            >
              {q}
            </button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="px-3 pb-4 pt-2">
        <div className="flex gap-2 items-end">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input); } }}
            placeholder="Ask about this analysis…"
            rows={2}
            className="flex-1 resize-none rounded-xl px-3 py-2.5 text-sm leading-relaxed"
            style={{
              background: 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.10)',
              color: '#e2e8f0',
              outline: 'none',
              fontFamily: 'inherit',
            }}
            onFocus={e => { e.target.style.borderColor = 'rgba(20,184,166,0.5)'; e.target.style.boxShadow = '0 0 0 3px rgba(20,184,166,0.12)'; }}
            onBlur={e => { e.target.style.borderColor = 'rgba(255,255,255,0.10)'; e.target.style.boxShadow = 'none'; }}
          />
          <button
            onClick={() => send(input)}
            disabled={!input.trim() || typing}
            className="btn-primary px-3 py-2 rounded-lg text-xs shrink-0"
          >
            ↑
          </button>
        </div>
        <div className="text-xs text-slate-700 mt-1">Enter to send · Shift+Enter for newline</div>
      </div>
    </div>
  );
}
