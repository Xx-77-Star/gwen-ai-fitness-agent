import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { test } from 'node:test';

const source = readFileSync(new URL('../frontend/app.js', import.meta.url), 'utf8');
const chatSource = source.slice(source.indexOf('const loadingState ='), source.indexOf('function normalizeTrace('));
const storage = () => {
  const data = new Map();
  return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value), removeItem: key => data.delete(key) };
};
function page(sessionStorage, localStorage, turns, historyFetch) {
  const elements = new Map();
  function element() {
    return { hidden: false, disabled: false, value: '', children: [], handlers: {},
      addEventListener(name, fn) { this.handlers[name] = fn; },
      append(node) { node.parent = this; this.children.push(node); },
      querySelectorAll() { return this.children; },
      focus() {}, remove() { this.parent.children = this.parent.children.filter(child => child !== this); },
    };
  }
  const document = { querySelector(selector) {
    if (!elements.has(selector)) elements.set(selector, element());
    return elements.get(selector);
  }, createElement: element };
  const sandbox = { document, sessionStorage, localStorage, crypto: { randomUUID: () => 'conv-test' },
    loadUserCity() {}, openAgentStatusStream() {}, closeAgentStatusStream() {}, hideAgentStatus() {},
    renderTrace() {}, chatBusy: false, chatAvatar: null, setChatCompanionStatus() {}, updateLoadingOverlay() {},
    parseJsonResponse: response => response.json(),
    fetch: async (url, options) => {
      if (options?.method === 'POST') {
        const body = JSON.parse(options.body);
        turns.push({role: 'user', content: body.message}, {role: 'assistant', content: '测试回答'});
        return {ok: true, json: async () => ({conversation_id: body.conversation_id, response: '测试回答'})};
      }
      return historyFetch ? historyFetch() : {ok: true, json: async () => ({messages: [...turns]})};
    },
  };
  vm.runInNewContext(chatSource, sandbox);
  return elements;
}
const settle = () => new Promise(resolve => setImmediate(resolve));
test('sent messages survive page reload without being sent again', async () => {
  const session = storage(), local = storage(), turns = [];
  const first = page(session, local, turns);
  await settle();
  first.get('#message-input').value = '刷新回归测试';
  await first.get('#chat-form').handlers.submit({preventDefault() {}});
  assert.equal(first.get('#message-list').children.length, 2);
  const reloaded = page(session, local, turns);
  await settle();
  assert.deepEqual(reloaded.get('#message-list').children.map(node => node.textContent), ['刷新回归测试', '测试回答']);
  assert.equal(turns.length, 2);
});

test('new chat stays empty on refresh without deleting the old transcript', async () => {
  const session = storage(), local = storage();
  session.setItem('fitlife-conversation-id', 'old');
  const turns = [{role: 'user', content: '旧消息'}];
  const loaded = page(session, local, turns);
  await settle();
  loaded.get('#new-chat').handlers.click();
  assert.equal(loaded.get('#message-list').children.length, 0);
  const reloaded = page(session, local, turns);
  await settle();
  assert.equal(reloaded.get('#message-list').children.length, 0);
  assert.equal(turns.length, 1);
});

test('history failure offers retry and never sends while history is loading', async () => {
  const session = storage(), local = storage();
  session.setItem('fitlife-conversation-id', 'old');
  let fail = true;
  const loaded = page(session, local, [], async () => {
    if (fail) throw new Error('offline');
    return {ok: true, json: async () => ({messages: [{role: 'user', content: '恢复成功'}]})};
  });
  assert.equal(loaded.get('#send-button').disabled, true);
  await settle();
  assert.equal(loaded.get('#send-button').disabled, true);
  assert.match(loaded.get('#message-list').children[0].textContent, /不会被清除/);
  fail = false;
  loaded.get('#message-list').children[0].children[0].handlers.click();
  await settle();
  assert.deepEqual(loaded.get('#message-list').children.map(node => node.textContent), ['恢复成功']);
  assert.equal(loaded.get('#send-button').disabled, false);
});
