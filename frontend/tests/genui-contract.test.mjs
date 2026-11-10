import test from 'node:test';
import assert from 'node:assert/strict';
import { inspectUi, getUiSystemPrompt } from '../src/lib/verified-ui-contract.ts';

const sources=[{id:'S1',title:'Configuration guide',text:'Use CqServerConn.properties. Read the file before changing settings.'},{id:'S2',title:'Platform guide',text:'Linux and Windows have documented alternatives.'}];
const inspect=(program)=>inspectUi(program,sources);

test('existing approved text remains compatible',()=>{
 const r=inspect('root = Stack([AnswerText("Use the configuration file. [S1]")])');
 assert.equal(r.valid,true);assert.deepEqual(r.citation_source_ids,['S1']);
});
test('task-shaped components contribute every factual field to verification',()=>{
 const r=inspect(`root = Stack([AnswerCard("Configuration file", "Use CqServerConn.properties.", ["S1"]), Procedure("Review settings", [{title:"Read the file", detail:"Read before changing settings.", code:"cat CqServerConn.properties", language:"shell", sourceIds:["S1"]}]), Configuration("Documented fields", [{key:"File",value:"CqServerConn.properties",description:"Configuration file",sourceIds:["S1"]}]), Notice("Prerequisite", "Read the file first.", "note", ["S1"])])`);
 assert.equal(r.valid,true);
 for(const field of ['Configuration file','Review settings','Read before changing','cat CqServerConn.properties','Documented fields','Prerequisite'])assert.ok(r.visible_text.includes(field));
 assert.deepEqual(r.citation_source_ids,['S1']);
});
test('all tab content is checked including inactive alternatives',()=>{
 const r=inspect('root = Stack([Tabs([{label:"Linux",children:[AnswerCard("Linux alternative","Linux detail.",["S1"])]},{label:"Windows",children:[AnswerCard("Windows alternative","Hidden panel detail.",["S2"])]}])])');
 assert.equal(r.valid,true);assert.ok(r.visible_text.includes('Hidden panel detail.'));assert.deepEqual(r.citation_source_ids,['S1','S2']);
});
test('unknown source in a hidden tab invalidates the whole presentation',()=>{
 const r=inspect('root = Stack([Tabs([{label:"A",children:[AnswerText("Known [S1]")]},{label:"B",children:[AnswerCard("Bad","Unsupported.",["S99"])]}])])');
 assert.equal(r.valid,false);
});
test('unknown components and malformed structured fields are rejected',()=>{
 for(const program of ['root = Stack([MadeUp("x")])','root = Stack([Procedure("Steps", [{title:"x",detail:"y",sourceIds:[]}])])','root = Stack([Configuration("Fields",[{key:"x",value:3,sourceIds:["S1"]}])])','root = Stack([DataTable("Values",["A","B"],[["one"]])])'])assert.equal(inspect(program).valid,false,program);
});
test('actions and state are outside the answer component contract',()=>{
 for(const program of ['root = Stack([AnswerText(@OpenUrl("https://example.com"))])','$x = 1\nroot = Stack([AnswerText("text")])'])assert.equal(inspect(program).valid,false);
});
test('prompt selects structures by content and disallows invented operational facts',()=>{
 const prompt=getUiSystemPrompt();
 for(const component of ['AnswerCard','Procedure','Configuration','Notice','Tabs'])assert.ok(prompt.includes(component));
 assert.ok(prompt.includes('do not fabricate extra sections'));
 assert.ok(prompt.includes('must not be hidden'));
});


test('browser origin follows the request host when Next normalizes localhost',async()=>{
 const {isAllowedOrigin}=await import('../src/lib/request-origin.ts');
 const original=process.env.APP_ORIGIN;
 delete process.env.APP_ORIGIN;
 try {
  const request=(origin,host='127.0.0.1:3000')=>new Request('http://localhost:3000/api/ui-validate',{headers:{origin,host}});
  assert.equal(isAllowedOrigin(request('http://127.0.0.1:3000')),true);
  assert.equal(isAllowedOrigin(request('https://evil.example')),false);
  assert.equal(isAllowedOrigin(request('https://127.0.0.1:3000')),false);
  assert.equal(isAllowedOrigin(request('invalid-origin')),false);
  assert.equal(isAllowedOrigin(new Request('http://localhost:3000/api/ui-validate')),true);
  process.env.APP_ORIGIN='https://demo.example';
  assert.equal(isAllowedOrigin(request('http://127.0.0.1:3000')),false);
  assert.equal(isAllowedOrigin(request('https://demo.example')),true);
 } finally {
  if(original===undefined)delete process.env.APP_ORIGIN;
  else process.env.APP_ORIGIN=original;
 }
});
