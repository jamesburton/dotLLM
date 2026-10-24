import json,sys
from jinja2.sandbox import ImmutableSandboxedEnvironment
tpl_path,out=sys.argv[1],sys.argv[2]
class Raise(Exception): pass
def raise_exception(m): raise Raise(m)
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
# HF's tojson: json.dumps(ensure_ascii=False), no sorting
env.filters['tojson']=lambda x,ensure_ascii=False,indent=None,separators=None,sort_keys=False: json.dumps(x,ensure_ascii=ensure_ascii,indent=indent,separators=separators,sort_keys=sort_keys)
env.globals['raise_exception']=raise_exception
t=env.from_string(open(tpl_path,encoding='utf-8',newline='').read())
U=lambda c:{"role":"user","content":c}
A=lambda c,**k:{"role":"assistant","content":c,**k}
S=lambda c:{"role":"system","content":c}
tool={"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}
base={"bos_token":"<|endoftext|>","eos_token":"<|im_end|>"}
G={"add_generation_prompt":True}
cases=[
 ("user_only_default_thinking",{"messages":[U("What is 2 + 2?")],**G}),
 ("user_only_thinking_off",{"messages":[U("What is 2 + 2?")],**G,"enable_thinking":False}),
 ("user_only_thinking_explicit_on",{"messages":[U("What is 2 + 2?")],**G,"enable_thinking":True}),
 ("effort_low",{"messages":[U("hi")],**G,"reasoning_effort":"low"}),
 ("effort_medium",{"messages":[U("hi")],**G,"reasoning_effort":"medium"}),
 ("effort_xhigh",{"messages":[U("hi")],**G,"reasoning_effort":"xhigh"}),
 ("effort_invalid_raises",{"messages":[U("hi")],**G,"reasoning_effort":"bogus"}),
 ("system_and_user",{"messages":[S("You are terse."),U("Hello")],**G}),
 ("system_and_user_thinking_off",{"messages":[S("You are terse."),U("Hello")],**G,"enable_thinking":False}),
 ("multi_turn",{"messages":[U("Hi"),A("Hello there!"),U("Tell me a joke")],**G}),
 ("multi_turn_reasoning_content",{"messages":[U("Hi"),A("Hello!",reasoning_content="  greeting  "),U("Again")],**G}),
 ("multi_turn_preserve_thinking_false",{"messages":[U("Hi"),A("Hello!",reasoning_content="r"),U("Again")],**G,"preserve_thinking":False}),
 ("no_generation_prompt",{"messages":[U("Hi"),A("Hello!")],"add_generation_prompt":False}),
 ("unicode_and_whitespace",{"messages":[U("  café ☃ \n")],**G}),
 ("no_user_message_raises",{"messages":[S("only system")],**G}),
 ("tools_declared",{"messages":[U("Weather in Paris?")],**G,"tools":[tool]}),
 ("tool_call_roundtrip",{"messages":[U("Weather in Paris?"),A("",tool_calls=[{"type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}]),{"role":"tool","content":"sunny"},U("thanks")],**G,"tools":[tool]}),
]
res=[]
for n,ctx in cases:
    ctx={**base,**ctx}
    try: r={"name":n,"context":ctx,"expected":t.render(**ctx)}
    except Raise as e: r={"name":n,"context":ctx,"expectedError":str(e)}
    res.append(r)
json.dump(res,open(out,'w',encoding='utf-8',newline='\n'),ensure_ascii=False,indent=1)
print(len(res))
for c in res: print(c['name'], repr(c.get('expected',c.get('expectedError')))[:230])
