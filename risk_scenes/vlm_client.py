"""Minimal OpenAI-compatible / Gemini chat client; API keys are read from the environment variable named in vlm_models.json."""
import base64, io, json, os, re, time, urllib.error, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'vlm_models.json')

def image_url(img, fmt='JPEG'):
    from PIL import Image
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, fmt, quality=90)
    return f'data:image/{fmt.lower()};base64,' + base64.b64encode(buf.getvalue()).decode()

def user_message(text, images=()):
    content = [{'type': 'image_url', 'image_url': {'url': image_url(im)}} for im in images]
    return {'role': 'user', 'content': content + [{'type': 'text', 'text': text}]}

def _actions_array(text):
    found = None
    for found in re.finditer('"actions"\\s*:\\s*\\[', text):
        pass
    if not found:
        return None
    (start, depth, quoted, escaped) = (found.end() - 1, 0, False, False)
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            (escaped, quoted) = (False, quoted) if escaped else (c == '\\', c != '"')
        elif c == '"':
            quoted = True
        elif c in '[]':
            depth += 1 if c == '[' else -1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1], strict=False)
                except json.JSONDecodeError:
                    return None
    return None

def parse_json(text):
    (start, end) = (text.find('{'), text.rfind('}'))
    if 0 <= start < end:
        try:
            return json.loads(text[start:end + 1], strict=False)
        except json.JSONDecodeError:
            pass
    actions = _actions_array(text)
    if actions is not None:
        return {'actions': actions}
    start = text.find('{')
    while start >= 0:
        depth = 0
        for i in range(start, len(text)):
            depth += {'{': 1, '}': -1}.get(text[i], 0)
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1], strict=False)
                except json.JSONDecodeError:
                    break
        start = text.find('{', start + 1)
    return None

class Client:

    def __init__(self, name, temperature=None, max_tokens=None, replies=None, seed=None):
        self.name = name
        self.retried = 0
        if name == 'mock':
            self.replies = list(replies or ['{"decision": "refuse", "reason": "mock"}'])
            return
        cfg = json.load(open(MODELS))[name]
        self.url = cfg['base_url'].rstrip('/') + '/chat/completions'
        self.model = cfg['model']
        self.native = cfg.get('api') == 'gemini'
        self.thinking = cfg.get('thinking_level')
        if self.native:
            self.url = f"{cfg['base_url'].rstrip('/')}/models/{self.model}:generateContent"
        self.extra = dict(cfg.get('extra', {}))
        if temperature is not None:
            self.extra['temperature'] = temperature
        if max_tokens is not None:
            self.extra['max_completion_tokens'] = max_tokens
        if seed is not None:
            self.extra['seed'] = seed
        self.key = os.environ.get(cfg['key_env'])
        if not self.key:
            raise SystemExit(f"set {cfg['key_env']} for model {name}")

    def _gemini_body(self, messages):
        (body, contents) = ({}, [])
        for m in messages:
            if m['role'] == 'system':
                body['systemInstruction'] = {'parts': [{'text': m['content']}]}
                continue
            parts = []
            for c in [{'type': 'text', 'text': m['content']}] if isinstance(m['content'], str) else m['content']:
                if c['type'] == 'text':
                    parts.append({'text': c['text']})
                else:
                    (head, data) = c['image_url']['url'].split(',', 1)
                    parts.append({'inlineData': {'mimeType': head[len('data:'):].split(';')[0], 'data': data}})
            contents.append({'role': 'model' if m['role'] == 'assistant' else 'user', 'parts': parts})
        names = {'seed': 'seed', 'temperature': 'temperature', 'max_completion_tokens': 'maxOutputTokens'}
        config = {names[k]: v for (k, v) in self.extra.items() if k in names}
        if self.thinking:
            config['thinkingConfig'] = {'thinkingLevel': self.thinking}
        return dict(body, contents=contents, generationConfig=config)

    @staticmethod
    def _gemini_reply(out):
        cand = (out.get('candidates') or [{}])[0]
        text = ''.join((p.get('text', '') for p in cand.get('content', {}).get('parts', []) if not p.get('thought')))
        u = out.get('usageMetadata', {})
        return (text, dict(prompt_tokens=u.get('promptTokenCount', 0), completion_tokens=u.get('candidatesTokenCount', 0), thinking_tokens=u.get('thoughtsTokenCount', 0), total_tokens=u.get('totalTokenCount', 0)))

    def chat(self, messages, retries=3):
        if self.name == 'mock':
            return (self.replies.pop(0) if len(self.replies) > 1 else self.replies[0], {})
        body = json.dumps(self._gemini_body(messages) if self.native else dict(model=self.model, messages=messages, **self.extra)).encode()
        auth = {'x-goog-api-key': self.key} if self.native else {'Authorization': f'Bearer {self.key}'}
        if not getattr(self, '_in_filter_retry', False):
            self.filtered_tries = 0
        for attempt in range(retries):
            req = urllib.request.Request(self.url, body, {'Content-Type': 'application/json', **auth})
            try:
                with urllib.request.urlopen(req, timeout=float(os.environ.get('VLM_TIMEOUT', 180))) as r:
                    out = json.load(r)
                if self.native:
                    return self._gemini_reply(out)
                if self.name.startswith('claude') and out['choices'][0].get('finish_reason') == 'content_filter':
                    self.filtered = getattr(self, 'filtered', 0) + 1
                    if self.filtered_tries >= 7:
                        raise RuntimeError(f'{self.name}: content_filter')
                    self.filtered_tries += 1
                    time.sleep(2)
                    self._in_filter_retry = True
                    try:
                        return self.chat(messages, retries)
                    finally:
                        self._in_filter_retry = False
                return (out['choices'][0]['message'].get('content') or '', out.get('usage', {}))
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors='replace')[:500]
                if e.code not in (408, 409, 429, 500, 502, 503, 504) or attempt == retries - 1:
                    raise RuntimeError(f'{self.name}: HTTP {e.code}: {msg}')
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == retries - 1:
                    raise RuntimeError(f'{self.name}: {e}')
            self.retried += 1
            time.sleep(min(60, 2 ** attempt * 2))
