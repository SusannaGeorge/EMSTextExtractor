from pymetamap import MetaMap
from spacy.lang.en import English
from spacy.pipeline import EntityRuler
from spacy import displacy
import csv, os, re, spacy, scispacy, en_ner_bc5cdr_md,re,json, subprocess,tempfile, datetime
from collections import defaultdict

def get_transcripts(file_path):
	results=[]
	ipfile=[]
	ct=0
	for file in os.listdir(file_path):
		if file.endswith(".txt"):
			ipfile.append(file)
			ecic_conv=open(file_path+'/'+file, 'r').read()
			texts = ecic_conv.split('\n')
			texts = list(filter(None, texts))
			texts.pop(0)
			transcript = ' '.join(texts)
			transcript = transcript.replace("'", "")
			#.replace(". ", ".").replace("? ", "?")
			transcript = re.sub(r"\s*(\(\?\)|(\[\?\]))+\s*","",transcript, flags=re.IGNORECASE)
			transcript = re.sub(r"((amb)|(ambulance)|(dc\sems)|(ecic)|(siscom)|(syscom)|(medstar)|(pgcf)|(trooper\s\d*)|(medic\s*\d*))\:",\
			" ",transcript, flags=re.IGNORECASE)
			transcript = re.sub(r"((um)|(uh))[\,\-\.]*\s","",transcript, flags=re.IGNORECASE)
			results.append(transcript)
			ct+=1
		if ct==71:
			break
	return results,ipfile

def getspacy_pattern_matched_entities(text):
	nlp = English()
	ruler = EntityRuler(nlp, validate=True)
	med7 = spacy.load("en_core_med7_lg")
	patterns = [{"label": "AGE", "pattern": [{"TEXT" : {"REGEX": "(\d+)|([a-z]+)"}},\
				{"TEXT" : {"REGEX": "(year(s)*|yr(s)*|month(s)*)"}}, {"TEXT" : {"REGEX": "(old)*"}}]},
				{"label": "AGE GROUP", "pattern": [{"TEXT" : {"REGEX": "(infant)|(adult)|(teen)|(baby)|(toddler)"}}]},
				{"label": "GENDER", "pattern": [{"TEXT" : {"REGEX": "(M|m|Fem|fem)ale"}}]},
				{"label": "BP", "pattern": [{"TEXT" : {"REGEX": "(\d+)"}},\
									{"TEXT" : {"REGEX": "(over|\/|\-)"}},
									{"TEXT" : {"REGEX": "(\d+)"}}]}]
				
	ruler.add_patterns(patterns)
	nlp.add_pipe(ruler)
	nlp.add_pipe(nlp.create_pipe('sentencizer'))
	doc1 = nlp(text)
	doc2 = med7(text)
	ent1=[]
	ent2=[]
	list(set([ent1.append((ent.label_, ent.text)) for ent in doc1.ents]))
	list(set([ent2.append((ent.label_, ent.text)) for ent in doc2.ents]))
	
	return ent1, ent2

def refine_values(entities, text):
	if re.search(r'(\s[a-z\d]+\sand\s[a-z\d\s]+)(year|month)\s(old)',str(text).lower()):
			r=re.sub(r'(.*)((\s[a-z\d]+\sand\s[a-z\d\s]+)(year|month)\s(old))|(.*)',r'\2',text, flags =re.IGNORECASE)
			entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'AGE' + '\', \'' + r + '\')'
	if [ele for ele in ['boy','man'] if re.search(r'\\b'+ele, str(text), flags=re.IGNORECASE)]:
		if entities and str(entities)!='':
			entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'GENDER' + '\', \'male\')'
		else:
			entities='[(\'' + 'GENDER' + '\', \'male\')]'
	if [ele for ele in ['girl','lady','woman','mother'] if re.search(r'\\b'+ele, str(text), flags=re.IGNORECASE)]:
		if entities and str(entities)!='':
			entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'GENDER' + '\', \'female\')'
		else:
			entities='[(\'' + 'GENDER' + '\', \'female\')]'
	if 'pulse' in str(text).lower():
		r=re.sub(r'(.*)(((good|low|high)\spulse)|(\s(\d{3}|\d{2})[a-z\s]+(pulse))|((pulse\s)[a-z\s]*(\d{3}|\d{2})))|(.*)',r'\2',text, flags=re.IGNORECASE)
		if r and 'pulse ox' not in r and re.sub('[^0-9]+', '', r) !='':
			if entities and str(entities)!='':
				entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'PULSE' + '\', \''+re.sub('[^0-9]+', '', r)+'\')'
			else:
				entities='[(\'' + 'PULSE' + '\', \''+re.sub('[^0-9]+', '', r)+'\')]'
	if 'heart rate' in str(text).lower():
		r=re.sub(r'(.*)((heart\srate(\'s)?\s(is|was)+[a-z\-\s]+)(\d+)|((\d+)[a-z\s]+(heart\srate(\'s)?))|((heart\srate(\'s)?)([\sa-z])+(\d+)))|(.*)',r'\2',\
					text, flags=re.IGNORECASE)
		if r and re.sub('[^0-9]+', '', r) !='':
			if entities and str(entities)!='':
				entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'PULSE' + '\', \''+re.sub('[^0-9]+', '', r)+'\')'
			else:
				entities='[(\'' + 'PULSE' + '\', \''+re.sub('[^0-9]+', '', r)+'\')]'
	if 'respirations' in str(text).lower():
		r=re.sub(r'(.*)(((\d{3}|\d{2}|\d)[a-z\s]+(respirations|respiratory))|((respirations|respiratory)[\sa-z]+(\d{3}|\d{2}|\d)))|(.*)',r'\2',text, flags=re.IGNORECASE)
		if r and re.sub('[^0-9]+', '', r) !='':
			if entities and str(entities)!='':
				entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'RESP' + '\', \''+re.sub('[^0-9]+', '', r)+'\')'
			else:
				entities='[(\'' + 'RESP' + '\', \''+re.sub('[^0-9]+', '', r)+'\')]'
	if 'blood glucose levels' in str(text).lower():
		r=re.sub(r'(.*)((blood\sglucose\slevels)[\sa-z]+(\d+))|(.*)',r'\2',text, flags=re.IGNORECASE)
		if r and re.sub('[^0-9]+', '', r) !='':
			if entities and str(entities)!='':
				entities=str(entities).replace('[','').replace(']','')+ ', (\'' + 'B.G.L' + '\', \''+re.sub('[^0-9]+', '', r)+'\')'
			else:
				entities='[(\'' + 'B.G.L' + '\', \''+re.sub('[^0-9]+', '', r)+'\')]'
	if re.search(r'((100|\d{2})%)|((100|\d{2})\spercent(age)?)',str(text), flags=re.IGNORECASE) or 'pulse ox' in str(text).lower():
		r=re.sub(r'(.*)\s(((100|\d{2})%)|((100|\d{2})\spercent(age)?)|((pulse\sox)|(pulseox))[\sa-z]+(\d+))|(.*)',r'\2',text, flags=re.IGNORECASE)
		if r and re.sub('[^0-9]+', '', r) !='':
			if entities and str(entities)!='':
				entities= str(entities).replace('[','').replace(']','')+', (\''+'SPO2'+'\', \''+ re.sub('[^0-9]+', '', r) +'\')'
			else:
				entities='[(\'' + 'SPO2'+'\', \''+ re.sub('[^0-9]+', '', r) +'\')]'
	if re.search(r'gcs',str(text), flags=re.IGNORECASE):
		r=re.sub(r'(.*)(gcs[a-z\,\s]+(1[0-5]|[3-9])|\s(1[0-5]|[3-9])[a-z\,\s]+gcs)|(.*)',r'\2',text, flags=re.IGNORECASE)
		if r and re.sub('[^0-9]+', '', r) !='':
			if entities and str(entities)!='':
				entities= str(entities).replace('[','').replace(']','')+', (\''+'GCS'+'\', \''+ re.sub('[^0-9]+', '', r) +'\')'
			else:
				entities='[(\''+'GCS'+'\', \''+ re.sub('[^0-9]+', '', r) +'\')]'
	# if re.search(r'mental\sstatus',str(text), flags=re.IGNORECASE):
		# r=re.sub(r'(.*)((((completely\s)?alert\sand\soriented)|normal|altered|disoriented)[a-z\s\?\:\.]+mental\sstatus)|(.*)',r'\3',text, flags=re.IGNORECASE)
		# if not r or r == '':
			# r=re.sub(r'(.*)(mental\sstatus[a-z\s\?\:\.]+(((completely\s)?alert\sand\soriented)|normal|altered|disoriented))|(.*)',r'\3',text, flags=re.IGNORECASE)
		# if entities and str(entities)!='':
			# entities= str(entities).replace('[','').replace(']','')+', (\''+'MENTAL ST'+'\', \''+ r +'\')'
		# else:
			# entities='[(\''+'MENTAL ST'+'\', \''+ r +'\')]'
	return entities
	
def get_dependency(token):
	dvalue=''
	if 'NN' in token.tag_ and 'subj' in token.dep_:
		dvalue = token.head.text
		for c in token.head.children:
			if any(item in c.dep_ for item in ['conj','mod','obl','neg','compound']) and c.text not in dvalue:
				ckids = ' '.join(ck.text for ck in c.children if any(item in ck.dep_ for item in ['conj','mod','obl','neg','compound']))
				if ckids and c.text not in dvalue and ckids not in dvalue:
					dvalue = dvalue + ' ' + c.text + ' ' + ckids
				elif c.text not in dvalue:
					dvalue = dvalue + ' ' + c.text
		for ck in token.children:
			if any(item in ck.dep_ for item in ['conj','mod','obl','neg','compound']) and ck.text not in dvalue:
				dvalue = dvalue + ' ' + ck.text
	elif 'NN' in token.tag_ and token.dep_ == 'compound':
		dvalue = token.head.text
		if 'NN' in token.head.tag_ and token.head.dep_ == 'nsubj' and token.head.head.text not in dvalue:
			dvalue = dvalue + ' ' + token.head.head.text
	elif 'NN' in token.tag_ and 'VB' in token.head.tag_:
		for c in token.head.children:
			if any(item in c.dep_ for item in ['conj','mod','obl','neg','compound']):
				ckids = ' '.join(ck.text for ck in c.children if any(item in ck.dep_ for item in ['conj','mod','obl','neg','compound']))
				if ckids and c.text not in dvalue and ckids not in dvalue:
					dvalue = dvalue + ' ' + c.text + ' ' + ckids
				elif c.text not in dvalue:
					dvalue = dvalue + ' ' + c.text 
		for c in token.children:
			if any(item in c.dep_ for item in ['conj','mod','obl','neg','compound']):
				ckids = ' '.join(ck.text for ck in c.children if any(item in ck.dep_ for item in ['conj','mod','obl','neg','compound']))
				if ckids and c.text not in dvalue and ckids not in dvalue:
					dvalue = dvalue + ' ' + c.text + ' ' + ckids
				elif c.text not in dvalue:
					dvalue = dvalue + ' ' + c.text
	elif 'NN' in token.tag_ and 'NN' in token.head.tag_:
		dvalue = token.head.text
		for c in token.head.children:
			if any(item in c.dep_ for item in ['conj','mod','obl','neg','compound']):
				ckids = ' '.join(ck.text for ck in c.children if any(item in ck.dep_ for item in ['conj','mod','obl','neg','compound']))
				if ckids and c.text not in dvalue and ckids not in dvalue:
					dvalue = dvalue + ' ' + c.text + ' ' + ckids
				elif c.text not in dvalue:
					dvalue = dvalue + ' ' + c.text 
		for c in token.children:
			if any(item in c.dep_ for item in ['conj','mod','obl','neg','compound']):
				ckids = ' '.join(ck.text for ck in c.children if any(item in ck.dep_ for item in ['conj','mod','obl','neg','compound']))
				if ckids and c.text not in dvalue and ckids not in dvalue:
					dvalue = dvalue + ' ' + c.text + ' ' + ckids
				elif c.text not in dvalue:
					dvalue = dvalue + ' ' + c.text
	elif 'NN' in token.tag_:
		for c in token.children:
			if any(item in c.dep_ for item in ['conj','mod','obl','neg','compound']) and c.text not in dvalue:
				dvalue = dvalue + ' ' + c.text
	elif 'JJ' in token.tag_:
		if token.text.lower != 'mental':
			dvalue = token.text
		kids = ' '.join(ck.text for ck in token.children if ck.dep_ in ['conj','mod','obl','neg','compound'])
		if kids and dvalue not in kids:
			dvalue = dvalue + ' ' + kids
	elif token.dep == 'ROOT':
		dvalue = ' '.join(c.text for c in token.children)
	else:
		if token.text.lower != 'mental':
			dvalue = token.text
	return dvalue
	
def extract_semtype_phrase(semtype, values, output_list, ent3, text):
	original_phrase_list=[]
	temp=[]
	
	text = text.replace(',','.')
	split_text = text.split('.')
	
	bionlp = en_ner_bc5cdr_md.load()
	bionlp.add_pipe(bionlp.create_pipe('sentencizer'))
	
	for line in output_list:
		line=line.strip()
		if line:
			temp.append(line)
		if '['+semtype+']' in line and int("".join(filter(str.isdigit, line))) >= 800:
			for rline in temp[::-1]:
				if 'Phrase' in rline:
					rline = rline.strip(" ;").replace("Phrase:", "").replace("'","").strip()
					rline = re.sub('^%s' % ",", "", rline)
					rline = re.sub('%s$' % ",", "", rline)
					if values and any(detail in rline.lower() for detail in values):
						match ='.'.join([s for s in split_text if rline.lower() in s.lower()])
						if 'pupils' in match or 'eyes' in match:
							match = re.sub(r'(.*)((pupils|eyes)[\sa-z,]+)|(.*)',r'\2',match, flags=re.IGNORECASE)
						doc=bionlp(match)						
						for token in doc:
							if any(detail in token.text.lower() for detail in values):
								ent3[token.text.upper()] = get_dependency(token)
					
					if rline not in original_phrase_list:
						original_phrase_list.append(rline)
						break
	return original_phrase_list, ent3


def mmop_extractor(output, text):
	semtype_list=['Body Location or Region', 'Body Part, Organ, or Organ Component', \
				'Clinical Attribute','Clinical Drug', 'Disease or Syndrome', 'Finding', \
				'Injury or Poisoning', 'Medical Device', 'Sign or Symptom']
	
	semtype_details_dict={'Body Location or Region':['head','chin','neck','shoulder','arm','back','abdomen','torso','knee'],
	'Body Part, Organ, or Organ Component':['eyes','pupils','lung'], \
	'Clinical Attribute':['medications'],\
	'Clinical Drug':[], \
	'Disease or Syndrome':[], \
	'Finding':['awake','alert','conscious','disoriented', 'lethargic', 'confused','trauma','history','loc','allergies', \
	'pain', 'level of consciousness','loss of consciousness','pupils','injuries'], \
	'Injury or Poisoning':['trauma'], \
	'Medical Device':[], \
	'Sign or Symptom':['pain']}
	
	output_list = output.split("\n")
	
	semantic_dict={}
	ent3=defaultdict(dict)
	
	for semtype in semtype_list:
		phrase_list,ent3 = extract_semtype_phrase(semtype, semtype_details_dict[semtype], output_list, ent3, text)
		if phrase_list and len(phrase_list)>0:
			semantic_dict[semtype]=phrase_list
						
	return semantic_dict,ent3
	
def metamap_fetch(sentences, mm_location):
	sentences = sentences.replace(",",".")
	sentences = sentences.split('. ')
	sentences = [sentences]
	if sentences is not None:
		in_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
		for sentence in sentences:
			in_file.write(b'%r\n' % sentence)
	else:
		print("No input defined.")
	in_file.flush()
	
	out_file = tempfile.NamedTemporaryFile(mode="r", delete=False)
	
	command = [mm_location]
	command.append(in_file.name)
	command.append(out_file.name)
	command.append("-y")
	command.append("--negex")
	command.append("-u")
	command.append("--conj")
	#command.append("-z")
	
	run_metamap = subprocess.Popen(command, stdout=subprocess.PIPE)
	
	while run_metamap.poll() is None:
		stdout = str(run_metamap.stdout.readline())
		if 'ERROR' in stdout:
			run_metamap.terminate()
			error = stdout.rstrip()
	output = str(out_file.read())
	return output
		
def information_extractor(text, semantic_dict):
	
	entities, ent2 =getspacy_pattern_matched_entities(text.lower())	
	entities=[t for t in (set(tuple(i) for i in entities))]
	ent2=[t for t in (set(tuple(i) for i in ent2))]
	
	entities = refine_values(entities,text)
	
	ent_list=list(eval(str(entities)))
	ent_list2=list(eval(str(ent2)))
	
	keys = ["AGE", "GENDER", "PULSE", "BP", "RESP", "B.G.L", "SPO2", "GCS", "MENTAL ST", "PATIENT COND"\
			,"MEDICATION","ALLERGIES","PAST MEDICAL HISTORY","PAIN","TRAUMA","PUPILS","LUNG SOUNDS","VERBAL",\
			"INJURY","MECHANISM OF INJURY","COMPLAINT","TREATMENT","NOTES"]

	result = {}
	
	for (k, v) in ent_list:
		if k and v:
			result.setdefault(k, []).append(v)
	for key in keys:
		if key not in result.keys():
			result[key] = ['']
	result2 = {}
	for (k, v) in ent_list2:
		result2.setdefault(k, []).append(v)
	
	d2=dict(result2)
	dd = defaultdict(list)

	for d in (semantic_dict, d2): # you can list as many input dicts as you want here
		for key, value in d.items():
			dd[key]= list(set(dd[key] + value))
	bionlp = en_ner_bc5cdr_md.load()
	bionlp.add_pipe(bionlp.create_pipe('sentencizer'))
	return dict(result), dd, semantic_dict


def refine_entity(ent1,ent2,ent3, text):
	
	bionlp = en_ner_bc5cdr_md.load()
	bionlp.add_pipe(bionlp.create_pipe('sentencizer'))
	bionlp.add_pipe(bionlp.create_pipe('merge_entities'))
	bionlp.add_pipe(bionlp.create_pipe('merge_noun_chunks'))
	
	replaced_txt = text.replace(',','.')
	split_text = replaced_txt.split('.')
	if ent2['Injury or Poisoning']:
		for s in split_text:
			ent1['INJURY'] = ent1['INJURY'] + [x for x in ent2['Injury or Poisoning'] if x.lower() in s.lower() and 'vehicle' not in s.lower() and 'collision' not in s.lower()]
	if ent2['DRUG']:
		ent1['MEDICATION']=ent2['DRUG']	
	if ent2['Medical Device']:
		ent1['TREATMENT']=ent2['Medical Device']	
	dtemp={'COMPLAINT':['complain'], 'INJURY':['injury','injuries'], \
	'MECHANISM OF INJURY':['fell','fall','gunshot','struck','fire','attack', 'collision','assaut','stab'],\
	'LUNG SOUNDS':['lung','lungs'],'VERBAL':['confused','groggy'],'PATIENT COND':['stable','unstable','critical'],\
	'MENTAL ST':['awake','alert','disoriented','oriented', 'lethargic','conscious','unconscious','unresponsive','loc','mental'],\
	'TRAUMA':['TRAUMA'],'PUPILS':['PUPILS'],\
	'PAIN':['PAIN'],'PAST MEDICAL HISTORY':['HISTORY'],'ALLERGIES':['allergies']}
	for key, value in dtemp.items():
		for v in value:
			if ent3[v.upper()]:
				if ent1[key] and len(ent1[key])>0:
					ent1[key] = [x for x in ent1[key] if x and x.strip()]
					if ent3[v.upper()] not in ent1[key]:
						ent1[key] = list(set(ent1[key] + [ent3[v.upper()]]))
				else:
					ent1[key] = list(set([ent3[v.upper()]]))
			else:
				if key in ['MECHANISM OF INJURY' , 'COMPLAINT', 'MENTAL ST']:
					ent1[key] = ent1[key] + [f.strip() for f in ent2['Finding'] if re.compile(r'\b'+ v.lower()).search(f.lower()) and '?' not in f]
					ent1[key] = ent1[key] + [s.strip() for s in split_text if re.compile(r'\b'+ v.lower()).search(s.lower()) and '?' not in s]
					ent1[key] = list(set([x for x in ent1[key] if x and x.strip()]))					
				else:
					matching = '. '.join([s for s in split_text if v.lower() in s.lower() and '?' not in s])
					d2 = bionlp(matching)
					for token in d2:
						if v.lower() in token.text.lower():
							if key =='PATIENT COND':
								pc=['patient','everything','he','she'] 
								if token.head.text in pc or any(item.text in pc for item in token.children) or any(item.text in pc for item in token.head.children):
									ent1[key] = list(set(ent1[key] + [get_dependency(token)]))
									ent1[key] = [x for x in ent1[key] if x and x.strip()]
							else:
								ent1[key] = list(set(ent1[key] + [get_dependency(token)]))
								ent1[key] = [x for x in ent1[key] if x and x.strip()]
	
	complaints = ent1['COMPLAINT']+ent1['PAIN']
	ent1['COMPLAINT'] = [x for x in complaints if x]
	for key in ent1:
		temp=ent1[key]
		s=[]
		for e in ent1[key]:
			s = s + [e for t in temp if e!=t and e.lower() in t.lower() or '?' in e]
		ent1[key] = list(set(ent1[key]) - set(s))
	ent1['NOTES']=[text]	
	return ent1
	
def extract_concepts(text,ip,i):
		
	sent_text=[text]	
	
	mm_concepts2 = metamap_fetch(text,'./public_mm/bin/metamap20')
	semantic_dict,ent3 = mmop_extractor(mm_concepts2, text)
	
	ent1, ent2, semantic_dict =information_extractor(text, semantic_dict)
	
	ent1 = refine_entity(ent1,ent2,ent3, text)
	for key in ent1:
		if not ent1[key] or (ent1[key] and len(ent1[key]) == 0):
			ent1[key] = [""]
	
	refined_extracted_info = json.dumps(ent1, indent=4, ensure_ascii=False)
	extracted_info = json.dumps(ent2, indent=4, ensure_ascii=False)
	ent3=json.dumps(ent3, indent=4, ensure_ascii=False)
	
	op='sentence text\n'+text+'\n****************************************************\n'\
	+'Refined_extracted_info\n'+str(refined_extracted_info)
	#+'\n*************************************************\n\n\n'\
	#+'mm_concepts\n'+str(mm_concepts2)
	# +'extracted_info\n'+str(extracted_info)+'\n*************************************************\n'\
	# +'ent3\n'+str(ent3)+'\n************************************************************\n'\
	# +'output of metamap_3 semantic_dict\n'+str(semantic_dict)+'\n*************************************************\n'\
	
	eop = 'Refined_extracted_info\n'+str(refined_extracted_info)
	
	file_path='./concept_outputs/'+str(i)+'_op_'+ip
	with open(file_path, 'w') as opfile:
		opfile.write(op)
	
	file_path2='./extracted_outputs/op_'+ip
	with open(file_path2, 'w') as opfile2:
		opfile2.write(eop)
	
transcripts,ipfile=get_transcripts('./ecic_transcripts')
for i in range(0,len(transcripts)):
	print('\n')
	x=datetime.datetime.now()
	extract_concepts(transcripts[i],ipfile[i],i)
	y=datetime.datetime.now()
	print('\nTime taken  :  ', ((y-x).total_seconds())/60)
text_sample='55 year old male found unconscious  driver side passenger seat \
of his car. His wife reported that he snorted a line of heroin before  \
just prior to losing to consciousness. Patient originally presented \
unresponsive and pale with shallow ineffective respirations at a rate \
of about 5, with heart rate was 118. His blood pressure was 205 over 119. \
His blood glucose levels 126, his O2 saturations was were 94% patient. \
Required bag mask ventilation with attached oxygen, however after \
0.25 mg of naloxone intravenously, patient is now awake and breathing \
normally, with Improvement in Vital Signs and respiratory status and no \
longer needs supplemental oxygen.'
#extract_concepts(text_sample, '0001_sample.txt')
