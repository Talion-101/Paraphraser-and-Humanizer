import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import random
import re

# Try to import better-profanity for content filtering
try:
    from better_profanity import profanity
    PROFANITY_AVAILABLE = True
    profanity.load_censor_words()
except ImportError:
    PROFANITY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ParaphraserEngine:
    """
    A paraphrasing engine that uses various techniques to humanize text
    and avoid AI detection.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.synonym_cache = {}
        
        # Common words that should be preferred for natural, everyday language
        self.common_words = {
            'use', 'make', 'get', 'take', 'see', 'know', 'think', 'want',
            'give', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try',
            'leave', 'call', 'good', 'new', 'first', 'last', 'long', 'great',
            'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different',
            'small', 'large', 'next', 'early', 'young', 'important', 'public',
            'bad', 'able', 'help', 'show', 'hear', 'play', 'run', 'move', 'like',
            'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit',
            'stand', 'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn',
            'change', 'lead', 'understand', 'watch', 'follow', 'stop', 'create', 'speak',
            'read', 'allow', 'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer',
            'remember', 'love', 'consider', 'appear', 'buy', 'wait', 'serve', 'die',
            'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain',
            'suggest', 'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull',
            'break', 'receive', 'agree', 'support', 'hit', 'produce', 'eat', 'cover',
            'catch', 'draw', 'choose', 'cause', 'point', 'listen', 'realize', 'place',
            'close', 'force', 'achieve', 'seek', 'deal', 'fight', 'teach', 'enjoy',
            'grow', 'keep', 'begin', 'start', 'seem', 'help', 'talk', 'turn', 'start',
            'might', 'show', 'hear', 'play', 'run', 'move', 'live', 'believe',
            'bring', 'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay',
            'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead', 'understand',
            'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow', 'add',
            'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
            'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect',
            'build', 'stay', 'fall', 'cut', 'reach', 'kill', 'remain', 'suggest',
            'raise', 'pass', 'sell', 'require', 'report', 'decide', 'pull', 'break',
            'receive', 'agree', 'support', 'hit', 'produce', 'eat', 'cover', 'catch',
            'draw', 'choose', 'cause', 'point', 'listen', 'realize', 'place', 'close',
            'force', 'achieve', 'seek', 'deal', 'fight', 'teach', 'enjoy', 'grow',
            'keep', 'begin', 'start', 'seem', 'help', 'talk', 'turn', 'start'
        }
        
        # Advanced/overly complex words to avoid in output
        self.advanced_words = {
            'meliorate', 'elucidate', 'ameliorate', 'perplex', 'concatenate',
            'obfuscate', 'cogitate', 'perambulate', 'soliloquy', 'ostentatious',
            'pellucid', 'sesquipedalian', 'synecdoche', 'propitious', 'bucolic',
            'perfunctory', 'perspicacious', 'vituperative', 'sycophantic', 'ephemeral',
            'ubiquitous', 'juxtapose', 'dichotomy', 'paradigm', 'epistemological',
            'ontological', 'phenomenological', 'teleological', 'hermeneutical', 'dialectical'
        }
    
    def get_synonyms(self, word, pos):
        """Get synonyms for a word based on its part of speech.
        Prefers simpler, more common synonyms."""
        if word in self.synonym_cache:
            return self.synonym_cache[word]
        
        synonyms = []
        
        # Words that produce bad/offensive/archaic replacements - skip them entirely
        skip_words = {
            'paradigm', 'methodology', 'instantiate', 'utilize', 'facilitate',
            'approach', 'method', 'analysis', 'research', 'study', 'inquiry', 'enquiry',
            'group', 'focus', 'interview', 'observation', 'data', 'result',
            'understanding', 'knowledge', 'experience', 'perspective', 'outcome',
            'student', 'teacher', 'researcher', 'participant', 'learner',
            'education', 'learning', 'teaching', 'academic', 'achievement',
            'context', 'meaning', 'theory', 'concept', 'framework'
        }
        
        # Bad replacement words found in output - never use these
        bad_replacements = {
            'meliorate', 'elucidate', 'ameliorate', 'perplex', 'concatenate',
            'obfuscate', 'cogitate', 'perambulate', 'soliloquy', 'ostentatious',
            'pellucid', 'sesquipedalian', 'synecdoche', 'propitious', 'bucolic',
            'wads', 'nidus', 'rankness', 'motley', 'eruditeness', 'amorphous',
            'kinda', 'finis', 'kinship', 'coarse', 'mount', 'fighting',
            'reside', 'dwell', 'pedantic', 'ofttimes', 'sooner',
            'helot', 'serf', 'thrall', 'bondsman', 'racism', 'racist',
            'slur', 'epithet', 'derogatory', 'offensive', 'bigot',
            'slave', 'bondage', 'servitude', 'bondwoman', 'bondman',
            # Bad replacements from 100% intensity test
            'drill', 'bookman', 'inquire', 'phenomenon', 'feeler', 'decisive',
            'stress', 'version', 'dispute', 'exit', 'realism', 'find', 'canvas',
            'surmise', 'numeric', 'bod', 'sight', 'try', 'amend', 'run', 'omen',
            'pawn', 'rout', 'immanent', 'call', 'import', 'soul', 'conduct',
            'vulgar', 'admit', 'audience', 'radical', 'schoolroom', 'notice',
            'comprehend', 'adjust', 'fullness', 'player', 'forte', 'sensibility',
            'conflict', 'access', 'head', 'aim', 'seeking', 'exam', 'measure',
            'deal', 'search', 'process', 'get', 'live', 'adopt', 'inducive',
            'see', 'form', 'operation', 'schoolroom', 'muse', 'epistemic',
            'hire', 'database', 'appears', 'proficiency', 'fixation', 'let',
            'name', 'infer', 'bank', 'take', 'dynamic', 'use', 'read', 'sentience',
            'condition', 'rigor', 'eubstance', 'value', 'still', 'issue',
            'accent', 'rigour', 'elaborate', 'bill', 'raise', 'cogency', 'work',
            'primal', 'fix', 'timber', 'brainwave', 'line', 'ask', 'inert',
            'debar', 'alive', 'reading', 'summons', 'entire', 'societal', 'world',
            'model', 'science', 'free', 'term', 'intent', 'finding', 'hold',
            'conclusion', 'illation', 'tender', 'deem', 'eminence', 'possibly',
            'full', 'argument', 'rest', 'vantage', 'limitation', 'preciseness',
            'trend', 'treatment', 'program', 'still', 'nicety', 'case', 'gobs',
            'sealed', 'otherwise', 'ply', 'consequently', 'field', 'scholar',
            'sight', 'reply', 'know', 'last', 'complexness', 'mensurable',
            'educator', 'adopt', 'variety', 'act',
            # Additional bad replacements from second output
            'event', 'attack', 'praxis', 'image', 'upshot', 'premise', 'premiss',
            'remainder', 'speak', 'topic', 'prime', 'rationalist', 'note', 'assess',
            'canvass', 'expend', 'lesson', 'resume', 'trial', 'settle', 'quite',
            'realise', 'grouping', 'doings', 'notion', 'watching', 'instructor',
            'profusion', 'paw', 'query', 'outgrowth', 'trace', 'sizing', 'educatee',
            'lowly', 'target', 'shine', 'preeminence', 'tool', 'technique',
            'numerical', 'tie', 'key', 'rule', 'drift', 'need', 'shape', 'office',
            'substance', 'hit', 'cognizance', 'retainer', 'coming', 'similar',
            'apply', 'trustiness', 'root', 'phallus', 'deep', 'report', 'set',
            'essay', 'appeal', 'view', 'intact', 'metier', 'lector', 'residue',
            'reward', 'care', 'charm', 'shade', 'lots', 'elaborated', 'bit',
            'mogul', 'want', 'flux', 'formalize', 'valuate', 'expiation',
            'substantive', 'revalue', 'bosom', 'ism',
            # Additional non-academic words to avoid
            'vogue', 'vulgarize', 'assemblage', 'cat\'s-paw', 'rede', 'germ', 'swan',
            'derogate', 'appendage', 'racy', 'wee', 'overture', 'limpidity', 'palm',
            'shew', 'sure', 'aroused', 'king', 'lotion', 'rack', 'rife', 'dissent',
            'rivet', 'notably', 'universe', 'sketch', 'rootle', 'augur', 'mortal',
            'breadth', 'mutual', 'reflexion',
            # Additional problematic replacements that don't match original meaning
            'pattern', 'effect', 'preparation', 'scheme', 'prevailing', 'effrontery',
            'assembling', 'construe', 'vital', 'emphasise', 'pore', 'worthful', 'departure',
            'prefer', 'yield', 'kind', 'bear', 'examine', 'better', 'prove', 'position',
            'compare', 'variable', 'allot', 'belief', 'uncouth', 'observance', 'pupil',
            'conform', 'motion', 'assay', 'serve', 'mensuration', 'survive', 'come',
            'year', 'execution', 'differ', 'mull', 'decided', 'evidently', 'integrated',
            'include', 'regress', 'close', 'sampling', 'demand', 'toy', 'part', 'too',
            'consistence', 'validness', 'think', 'appraise', 'control', 'origin', 'survey',
            'liken', 'stay', 'void', 'Still', 'function', 'Nevertheless', 'hear', 'procedure',
            'fault', 'reality', 'degage', 'design', 'give', 'liberal', 'living', 'width',
            'vulgarise', 'advance', 'limit', 'plow', 'style', 'evaluate', 'seize', 'forge',
            'explicate', 'leave', 'tale', 'unveil', 'copy', 'clearly', 'sundry', 'realize',
            'answer', 'offer', 'amply', 'treat', 'imply', 'resultant', 'pedagog', 'mix',
            'hug', 'approaching', 'construe', 'assemble', 'depart', 'yield',
            # Additional archaic and inappropriate words to avoid
            'villein', 'epitome', 'sire', 'mensurate', 'presage', 'rendering',
            'surmisal', 'watch', 'ponder', 'hardiness', 'bailiwick', 'withal', 'espouse',
            'pedagogue',
            # Additional formal/academic words to avoid
            'utilize', 'facilitate', 'implement', 'demonstrate', 'illustrate',
            'constitute', 'establish', 'determine', 'evaluate', 'assess',
            'examine', 'investigate', 'analyze', 'explore', 'discover',
            'identify', 'recognize', 'acknowledge', 'appreciate', 'comprehend',
            'understand', 'perceive', 'discern', 'distinguish', 'differentiate',
            'distinguish', 'differentiate', 'discriminate', 'differentiate',
            'substantiate', 'corroborate', 'validate', 'verify', 'confirm',
            'authenticate', 'certify', 'attest', 'witness', 'testify',
            'articulate', 'express', 'convey', 'communicate', 'transmit',
            'disseminate', 'propagate', 'circulate', 'distribute', 'spread',
            'accumulate', 'amass', 'collect', 'gather', 'assemble',
            'congregate', 'aggregate', 'consolidate', 'integrate', 'incorporate',
            'merge', 'blend', 'combine', 'unite', 'join',
            'connect', 'link', 'associate', 'relate', 'correlate',
            'correspond', 'match', 'equate', 'parallel', 'resemble',
            'simulate', 'imitate', 'mimic', 'emulate', 'reproduce',
            'generate', 'produce', 'create', 'fabricate', 'manufacture',
            'construct', 'build', 'erect', 'establish', 'found',
            'originate', 'initiate', 'commence', 'begin', 'start',
            'terminate', 'conclude', 'finalize', 'complete', 'finish',
            'accomplish', 'achieve', 'attain', 'reach', 'realize',
            'acquire', 'obtain', 'secure', 'procure', 'gain',
            'retain', 'preserve', 'maintain', 'sustain', 'uphold',
            'support', 'assist', 'aid', 'help', 'serve',
            'benefit', 'advantage', 'profit', 'gain', 'improve',
            'enhance', 'augment', 'amplify', 'magnify', 'intensify',
            'strengthen', 'reinforce', 'fortify', 'consolidate', 'solidify',
            'weaken', 'diminish', 'reduce', 'decrease', 'lessen',
            'minimize', 'mitigate', 'alleviate', 'relieve', 'ease',
            'eliminate', 'remove', 'eradicate', 'extinguish', 'abolish',
            'abolish', 'annul', 'revoke', 'rescind', 'cancel',
            'neglect', 'ignore', 'disregard', 'overlook', 'dismiss',
            'reject', 'refuse', 'decline', 'deny', 'repudiate',
            'oppose', 'resist', 'challenge', 'confront', 'face',
            'encounter', 'meet', 'confront', 'face', 'brave',
            'endure', 'withstand', 'tolerate', 'bear', 'suffer',
            'experience', 'undergo', 'encounter', 'face', 'meet',
            'participate', 'engage', 'involve', 'include', 'incorporate',
            'comprise', 'consist', 'constitute', 'compose', 'form',
            'represent', 'symbolize', 'signify', 'denote', 'indicate',
            'suggest', 'imply', 'infer', 'deduce', 'conclude',
            'assume', 'presume', 'suppose', 'surmise', 'guess',
            'speculate', 'hypothesize', 'theorize', 'postulate', 'propose',
            'recommend', 'advise', 'suggest', 'propose', 'advocate',
            'encourage', 'inspire', 'motivate', 'stimulate', 'prompt',
            'persuade', 'convince', 'influence', 'sway', 'affect',
            'impact', 'affect', 'influence', 'shape', 'mold',
            'transform', 'convert', 'change', 'alter', 'modify',
            'adapt', 'adjust', 'accommodate', 'conform', 'suit',
            'suit', 'fit', 'match', 'correspond', 'align',
            'coordinate', 'organize', 'arrange', 'order', 'structure',
            'systematize', 'standardize', 'regulate', 'control', 'manage',
            'supervise', 'oversee', 'monitor', 'observe', 'watch',
            'inspect', 'examine', 'scrutinize', 'analyze', 'study',
            'investigate', 'explore', 'research', 'inquire', 'question',
            'interrogate', 'interview', 'survey', 'poll', 'canvass',
            'assess', 'evaluate', 'judge', 'rate', 'score',
            'estimate', 'calculate', 'compute', 'reckon', 'figure',
            'predict', 'forecast', 'anticipate', 'expect', 'foresee',
            'project', 'envision', 'imagine', 'visualize', 'picture',
            'design', 'plan', 'devise', 'conceive', 'create',
            'develop', 'evolve', 'progress', 'advance', 'proceed',
            'continue', 'persist', 'endure', 'last', 'remain',
            'survive', 'live', 'exist', 'subsist', 'endure',
            'flourish', 'thrive', 'prosper', 'succeed', 'prevail',
            'dominate', 'prevail', 'rule', 'govern', 'control',
            'govern', 'rule', 'reign', 'dominate', 'command',
            'command', 'order', 'direct', 'instruct', 'guide',
            'lead', 'guide', 'direct', 'steer', 'pilot',
            'follow', 'pursue', 'chase', 'track', 'trace',
            'search', 'seek', 'look', 'hunt', 'find',
            'discover', 'find', 'locate', 'identify', 'detect',
            'reveal', 'disclose', 'uncover', 'expose', 'unveil',
            'conceal', 'hide', 'cover', 'mask', 'veil',
            'protect', 'defend', 'guard', 'shield', 'safeguard',
            'preserve', 'conserve', 'save', 'rescue', 'deliver',
            'liberate', 'free', 'release', 'discharge', 'dismiss',
            'employ', 'use', 'utilize', 'apply', 'exploit',
            'waste', 'squander', 'misuse', 'abuse', 'misapply',
            'damage', 'harm', 'hurt', 'injure', 'wound',
            'repair', 'fix', 'mend', 'restore', 'renew',
            'improve', 'better', 'enhance', 'upgrade', 'refine',
            'worsen', 'deteriorate', 'decline', 'degenerate', 'decay',
            'grow', 'develop', 'expand', 'increase', 'multiply',
            'shrink', 'contract', 'decrease', 'reduce', 'diminish',
            'rise', 'ascend', 'climb', 'mount', 'soar',
            'fall', 'drop', 'descend', 'plunge', 'crash',
            'move', 'shift', 'transfer', 'relocate', 'displace',
            'transport', 'carry', 'convey', 'transfer', 'move',
            'send', 'transmit', 'dispatch', 'deliver', 'convey',
            'receive', 'get', 'obtain', 'acquire', 'gain',
            'accept', 'receive', 'take', 'admit', 'acknowledge',
            'reject', 'refuse', 'decline', 'deny', 'spurn',
            'choose', 'select', 'pick', 'opt', 'decide',
            'prefer', 'favor', 'like', 'enjoy', 'love',
            'hate', 'dislike', 'loathe', 'detest', 'despise',
            'fear', 'dread', 'terror', 'horror', 'panic',
            'hope', 'wish', 'desire', 'want', 'need',
            'believe', 'trust', 'faith', 'confidence', 'reliance',
            'doubt', 'suspect', 'question', 'challenge', 'dispute',
            'know', 'understand', 'comprehend', 'grasp', 'apprehend',
            'learn', 'study', 'teach', 'educate', 'train',
            'teach', 'instruct', 'educate', 'train', 'coach',
            'read', 'write', 'speak', 'listen', 'hear',
            'say', 'tell', 'speak', 'state', 'declare',
            'ask', 'question', 'inquire', 'query', 'interrogate',
            'answer', 'respond', 'reply', 'retort', 'counter',
            'agree', 'consent', 'approve', 'accept', 'endorse',
            'disagree', 'dissent', 'object', 'oppose', 'protest',
            'promise', 'pledge', 'vow', 'swear', 'guarantee',
            'threaten', 'menace', 'intimidate', 'coerce', 'force',
            'help', 'assist', 'aid', 'support', 'serve',
            'hurt', 'harm', 'damage', 'injure', 'wound',
            'heal', 'cure', 'treat', 'remedy', 'restore',
            'kill', 'murder', 'slay', 'execute', 'assassinate',
            'die', 'perish', 'expire', 'decease', 'pass',
            'live', 'survive', 'exist', 'subsist', 'endure',
            'born', 'created', 'produced', 'generated', 'formed',
            'begin', 'start', 'commence', 'initiate', 'launch',
            'end', 'finish', 'complete', 'conclude', 'terminate',
            'stop', 'halt', 'cease', 'quit', 'discontinue',
            'go', 'proceed', 'advance', 'progress', 'continue',
            'come', 'arrive', 'reach', 'approach', 'near',
            'stay', 'remain', 'wait', 'abide', 'linger',
            'leave', 'depart', 'go', 'exit', 'withdraw',
            'enter', 'come', 'go', 'access', 'penetrate',
            'exit', 'leave', 'depart', 'withdraw', 'retreat',
            'open', 'close', 'shut', 'lock', 'unlock',
            'fasten', 'secure', 'attach', 'connect', 'join',
            'loosen', 'release', 'detach', 'disconnect', 'separate',
            'break', 'smash', 'crush', 'shatter', 'fracture',
            'build', 'construct', 'create', 'make', 'form',
            'destroy', 'demolish', 'ruin', 'wreck', 'devastate',
            'clean', 'wash', 'scrub', 'polish', 'shine',
            'dirty', 'soil', 'stain', 'spot', 'mark',
            'buy', 'purchase', 'acquire', 'obtain', 'get',
            'sell', 'market', 'trade', 'exchange', 'deal',
            'give', 'donate', 'grant', 'bestow', 'present',
            'take', 'seize', 'grab', 'snatch', 'capture',
            'steal', 'rob', 'thieve', 'pilfer', 'purloin',
            'share', 'divide', 'split', 'separate', 'part',
            'keep', 'hold', 'retain', 'maintain', 'preserve',
            'lose', 'misplace', 'drop', 'forget', 'mislay',
            'find', 'discover', 'locate', 'uncover', 'reveal',
            'hide', 'conceal', 'cover', 'mask', 'veil',
            'show', 'display', 'exhibit', 'present', 'demonstrate',
            'see', 'look', 'watch', 'observe', 'view',
            'hear', 'listen', 'attend', 'heed', 'notice',
            'feel', 'touch', 'sense', 'perceive', 'experience',
            'smell', 'scent', 'odor', 'fragrance', 'aroma',
            'taste', 'flavor', 'savor', 'relish', 'enjoy',
            'think', 'consider', 'ponder', 'reflect', 'meditate',
            'remember', 'recall', 'recollect', 'reminisce', 'reflect',
            'forget', 'lose', 'misplace', 'overlook', 'neglect',
            'dream', 'imagine', 'fantasize', 'envision', 'visualize',
            'wake', 'arise', 'awaken', 'rouse', 'stir',
            'sleep', 'rest', 'slumber', 'nap', 'doze',
            'eat', 'consume', 'devour', 'ingest', 'swallow',
            'drink', 'sip', 'gulp', 'quaff', 'imbibe',
            'laugh', 'chuckle', 'giggle', 'snicker', 'cackle',
            'cry', 'weep', 'sob', 'wail', 'bawl',
            'smile', 'grin', 'beam', 'smirk', 'simper',
            'frown', 'scowl', 'grimace', 'pout', 'sulk',
            'shout', 'yell', 'scream', 'shriek', 'bellow',
            'whisper', 'murmur', 'mutter', 'mumble', 'murmur',
            'sing', 'chant', 'carol', 'hum', 'croon',
            'dance', 'prance', 'skip', 'hop', 'jump',
            'run', 'sprint', 'dash', 'rush', 'race',
            'walk', 'stroll', 'saunter', 'wander', 'roam',
            'sit', 'seat', 'rest', 'settle', 'perch',
            'stand', 'rise', 'mount', 'ascend', 'climb',
            'lie', 'recline', 'rest', 'lounge', 'relax',
            'fall', 'drop', 'plummet', 'tumble', 'topple',
            'rise', 'ascend', 'climb', 'mount', 'soar',
            'fly', 'soar', 'glide', 'float', 'hover',
            'swim', 'dive', 'plunge', 'submerge', 'immerse',
            'drive', 'ride', 'travel', 'journey', 'voyage',
            'work', 'labor', 'toil', 'strive', 'struggle',
            'play', 'game', 'sport', 'fun', 'recreation',
            'fight', 'battle', 'war', 'combat', 'conflict',
            'peace', 'calm', 'quiet', 'silence', 'stillness',
            'love', 'hate', 'like', 'dislike', 'feel',
            'happy', 'sad', 'glad', 'sorry', 'joyful',
            'angry', 'mad', 'furious', 'irate', 'enraged',
            'afraid', 'scared', 'frightened', 'terrified', 'petrified',
            'brave', 'courageous', 'bold', 'fearless', 'daring',
            'coward', 'timid', 'fearful', 'anxious', 'nervous',
            'strong', 'weak', 'powerful', 'helpless', 'mighty',
            'big', 'small', 'large', 'tiny', 'huge',
            'fast', 'slow', 'quick', 'rapid', 'swift',
            'hot', 'cold', 'warm', 'cool', 'freezing',
            'good', 'bad', 'better', 'worse', 'best',
            'right', 'wrong', 'correct', 'incorrect', 'true',
            'false', 'real', 'fake', 'genuine', 'authentic',
            'new', 'old', 'young', 'ancient', 'modern',
            'first', 'last', 'beginning', 'end', 'middle',
            'high', 'low', 'tall', 'short', 'deep',
            'wide', 'narrow', 'broad', 'thin', 'thick',
            'heavy', 'light', 'dark', 'bright', 'clear',
            'clean', 'dirty', 'pure', 'impure', 'fresh',
            'hard', 'soft', 'rough', 'smooth', 'sharp',
            'dull', 'blunt', 'pointed', 'round', 'square',
            'long', 'short', 'brief', 'extended', 'stretched',
            'rich', 'poor', 'wealthy', 'impoverished', 'affluent',
            'smart', 'stupid', 'intelligent', 'dumb', 'clever',
            'wise', 'foolish', 'sensible', 'ridiculous', 'absurd',
            'beautiful', 'ugly', 'pretty', 'hideous', 'attractive',
            'important', 'trivial', 'significant', 'insignificant', 'major',
            'easy', 'difficult', 'simple', 'complex', 'hard',
            'safe', 'dangerous', 'secure', 'risky', 'hazardous',
            'certain', 'uncertain', 'sure', 'unsure', 'definite',
            'possible', 'impossible', 'probable', 'improbable', 'likely',
            'always', 'never', 'sometimes', 'often', 'rarely',
            'yes', 'no', 'maybe', 'perhaps', 'possibly',
            'here', 'there', 'everywhere', 'nowhere', 'somewhere',
            'now', 'then', 'later', 'soon', 'never',
            'today', 'tomorrow', 'yesterday', 'tonight', 'morning',
            'afternoon', 'evening', 'night', 'day', 'week',
            'month', 'year', 'decade', 'century', 'millennium',
            'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'ten',
            'hundred', 'thousand', 'million', 'billion', 'trillion',
            'first', 'second', 'third', 'fourth', 'fifth',
            'sixth', 'seventh', 'eighth', 'ninth', 'tenth',
            'all', 'none', 'some', 'many', 'few',
            'much', 'little', 'more', 'less', 'most',
            'least', 'both', 'either', 'neither', 'each',
            'every', 'any', 'some', 'no', 'none',
            'who', 'what', 'where', 'when', 'why',
            'how', 'which', 'whose', 'whom', 'this',
            'that', 'these', 'those', 'it', 'they',
            'he', 'she', 'we', 'you', 'i',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'her', 'its',
            'our', 'their', 'mine', 'yours', 'hers',
            'ours', 'theirs', 'myself', 'yourself', 'himself',
            'herself', 'itself', 'ourselves', 'themselves', 'each',
            'other', 'another', 'others', 'else', 'same',
            'different', 'similar', 'alike', 'unlike', 'equal',
            'better', 'worse', 'best', 'worst', 'good',
            'bad', 'well', 'ill', 'fine', 'okay',
            'so', 'very', 'too', 'quite', 'rather',
            'just', 'only', 'even', 'still', 'already',
            'also', 'too', 'as', 'well', 'either',
            'neither', 'nor', 'both', 'and', 'but',
            'or', 'yet', 'so', 'for', 'because',
            'since', 'as', 'though', 'although', 'even',
            'if', 'unless', 'until', 'while', 'when',
            'where', 'whether', 'before', 'after', 'during',
            'without', 'with', 'by', 'from', 'of',
            'in', 'on', 'at', 'to', 'into',
            'onto', 'upon', 'over', 'under', 'above',
            'below', 'between', 'among', 'through', 'throughout',
            'across', 'along', 'around', 'behind', 'before',
            'after', 'beside', 'beyond', 'near', 'far',
            'here', 'there', 'everywhere', 'nowhere', 'somewhere',
            'up', 'down', 'out', 'off', 'away',
            'back', 'forward', 'backward', 'sideways', 'around',
            'again', 'once', 'twice', 'always', 'never',
            'sometimes', 'often', 'rarely', 'seldom', 'usually',
            'generally', 'typically', 'normally', 'naturally', 'certainly',
            'definitely', 'probably', 'possibly', 'maybe', 'perhaps',
            'please', 'thank', 'thanks', 'sorry', 'excuse',
            'hello', 'hi', 'goodbye', 'bye', 'farewell',
            'yes', 'no', 'okay', 'alright', 'sure',
            'fine', 'good', 'great', 'excellent', 'wonderful',
            'terrible', 'awful', 'horrible', 'bad', 'poor'
        }
        
        # Skip words that are already fine as-is
        if word.lower() in skip_words:
            self.synonym_cache[word] = []
            return []
        
        # Map POS tags to wordnet POS
        pos_mapping = {
            'NN': wordnet.NOUN,
            'NNS': wordnet.NOUN,
            'VB': wordnet.VERB,
            'VBD': wordnet.VERB,
            'VBG': wordnet.VERB,
            'VBN': wordnet.VERB,
            'VBP': wordnet.VERB,
            'VBZ': wordnet.VERB,
            'JJ': wordnet.ADJ,
            'JJR': wordnet.ADJ,
            'JJS': wordnet.ADJ,
            'RB': wordnet.ADV,
            'RBR': wordnet.ADV,
            'RBS': wordnet.ADV,
        }
        
        wordnet_pos = pos_mapping.get(pos)
        
        if wordnet_pos:
            for synset in wordnet.synsets(word, pos=wordnet_pos):
                for lemma in synset.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    synonym_lower = synonym.lower()
                    
                    # Enhanced filter criteria - prefer simpler, common words
                    if (synonym_lower != word.lower() and 
                        synonym_lower not in skip_words and
                        synonym_lower not in bad_replacements and
                        len(synonym) <= len(word) + 3 and  # Similar length - keep it short
                        len(synonym) < 11 and  # Max 10 chars (plain, simple words)
                        ' ' not in synonym and  # Single words only
                        len(synonym) > 2 and  # At least 3 chars
                        # Filter out archaic/formal patterns
                        not synonym_lower.endswith('ate') and  # Avoid formal -ate words
                        not synonym_lower.endswith('ize') and  # Avoid formal -ize words
                        not synonym_lower.endswith('ify') and  # Avoid formal -ify words
                        not synonym_lower.endswith('tion') and  # Avoid formal -tion words
                        not synonym_lower.endswith('sion') and  # Avoid formal -sion words
                        not synonym_lower.endswith('ment') and  # Avoid formal -ment words
                        not synonym_lower.endswith('ness') and  # Avoid formal -ness words
                        not synonym_lower.endswith('ity') and  # Avoid formal -ity words
                        not synonym_lower.endswith('ance') and  # Avoid formal -ance words
                        not synonym_lower.endswith('ence') and  # Avoid formal -ence words
                        not synonym_lower.endswith('ant') and  # Avoid formal -ant words
                        not synonym_lower.endswith('ent') and  # Avoid formal -ent words
                        not synonym_lower.endswith('al') and  # Avoid formal -al words
                        not synonym_lower.endswith('ic') and  # Avoid formal -ic words
                        not synonym_lower.endswith('ive') and  # Avoid formal -ive words
                        not synonym_lower.endswith('ful') and  # Avoid formal -ful words
                        not synonym_lower.endswith('ous') and  # Avoid formal -ous words
                        not synonym_lower.endswith('able') and  # Avoid formal -able words
                        not synonym_lower.endswith('ible') and  # Avoid formal -ible words
                        # Filter out uncommon letter patterns
                        'ae' not in synonym_lower and  # Archaic pattern
                        'oe' not in synonym_lower and  # Archaic pattern
                        'ph' not in synonym_lower and  # Greek-derived (often formal)
                        'rh' not in synonym_lower and  # Greek-derived (often formal)
                        'pt' not in synonym_lower and  # Greek-derived (often formal)
                        'sc' not in synonym_lower and  # Uncommon pattern
                        'sch' not in synonym_lower and  # Uncommon pattern
                        'ch' not in synonym_lower and  # Uncommon pattern
                        'th' not in synonym_lower and  # Uncommon pattern
                        'wh' not in synonym_lower and  # Uncommon pattern
                        'qu' not in synonym_lower and  # Uncommon pattern
                        'x' not in synonym_lower and  # Uncommon letter
                        'z' not in synonym_lower and  # Uncommon letter
                        # Filter out words with 3+ consecutive consonants
                        not any(c1.isalpha() and c2.isalpha() and c3.isalpha() and 
                               c1 not in 'aeiou' and c2 not in 'aeiou' and c3 not in 'aeiou'
                               for c1, c2, c3 in zip(synonym_lower, synonym_lower[1:], synonym_lower[2:]))):
                        synonyms.append(synonym)
        
        # Prioritize common words - sort with common words first, then by length
        def sort_key(s):
            is_common = s.lower() in self.common_words
            # Common words first (False=0, True=1, so we want False first), then by length
            return (not is_common, len(s))
        
        synonyms = sorted(list(set(synonyms)), key=sort_key)
        # Limit to 3 best options - prioritize common words
        synonyms = synonyms[:3]
        self.synonym_cache[word] = synonyms
        return synonyms
    
    def join_tokens_properly(self, tokens):
        """Join tokens while keeping punctuation attached to previous words.
        Fixes spacing around quotes and apostrophes."""
        if not tokens:
            return ""
        
        result = []
        punctuation = '.,;:!?)-'
        quote_marks = '\'"'
        
        for i, token in enumerate(tokens):
            if i == 0:
                result.append(token)
            elif token in punctuation or (len(token) > 0 and token[0] in punctuation):
                # No space before punctuation
                result[-1] += token
            elif token in quote_marks or (len(token) > 0 and token[0] in quote_marks):
                # Handle quotes: attach to previous word if closing, separate if opening
                if token in ['"', "'"] and i > 0:
                    # Single quote or double quote - if it's likely closing quote, attach
                    result[-1] += token
                else:
                    result.append(token)
            else:
                result.append(token)
        
        # Post-process to fix spacing around quotes and apostrophes
        final_text = ' '.join(result)
        # Fix space before closing quotes/apostrophes: "word ' " -> "word'"
        final_text = final_text.replace(" '", "'")
        final_text = final_text.replace(' "', '"')
        # Fix space after opening quotes
        final_text = final_text.replace('" ', '"')
        final_text = final_text.replace("' ", "'")
        
        return final_text
    
    def replace_with_synonyms(self, text, intensity=0.5):
        """Replace words with synonyms based on intensity."""
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            paraphrased_tokens = []
            for word, pos in pos_tags:
                # Skip punctuation and stop words with lower probability
                if word.lower() in self.stop_words or not word.isalpha() or len(word) < 4:
                    paraphrased_tokens.append(word)
                else:
                    # Increased replacement intensity with quality filters
                    if random.random() < intensity:  # Direct intensity mapping
                        synonyms = self.get_synonyms(word, pos)
                        if synonyms:
                            replacement = random.choice(synonyms)
                            paraphrased_tokens.append(replacement)
                        else:
                            paraphrased_tokens.append(word)
                    else:
                        paraphrased_tokens.append(word)
            
            paraphrased_sentences.append(self.join_tokens_properly(paraphrased_tokens))
        
        return ' '.join(paraphrased_sentences)
    
    def restructure_sentences(self, text):
        """Restructure sentences to vary sentence patterns."""
        sentences = sent_tokenize(text)
        restructured = []
        
        for sentence in sentences:
            # Simple restructuring by moving clauses or adding variations
            sentence = sentence.strip()
            if sentence.endswith('.'):
                sentence = sentence[:-1]
            
            # Add variations
            if random.random() < 0.3 and len(sentence) > 20:
                # Sometimes restructure by moving subject
                tokens = word_tokenize(sentence)
                if len(tokens) > 5:
                    # Shuffle some middle words (but not the beginning structure too much)
                    mid_point = len(tokens) // 2
                    if mid_point > 2:
                        restructured.append(sentence + '.')
                    else:
                        restructured.append(sentence + '.')
                else:
                    restructured.append(sentence + '.')
            else:
                restructured.append(sentence + '.')
        
        return ' '.join(restructured)
    
    def add_variations(self, text):
        """Add minor grammatical variations."""
        # Replace common contractions with expanded forms and vice versa
        contractions = {
            "don't": "do not",
            "doesn't": "does not",
            "didn't": "did not",
            "won't": "will not",
            "wouldn't": "would not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "haven't": "have not",
            "hasn't": "has not",
            "hadn't": "had not",
            "it's": "it is",
            "that's": "that is",
            "what's": "what is",
            "who's": "who is",
        }
        
        result = text
        for contraction, expanded in contractions.items():
            # Replace contractions with some probability
            if random.random() < 0.4:
                result = re.sub(r'\b' + contraction + r'\b', expanded, result, flags=re.IGNORECASE)
        
        return result
    
    def filter_content(self, text):
        """
        Filter output text to remove vulgar/racist content and advanced words.
        Uses better-profanity if available, plus custom filters.
        
        Args:
            text: Text to filter
            
        Returns:
            Cleaned text safe for academic use
        """
        if not text:
            return text
        
        result = text
        
        # Filter using better-profanity if available
        if PROFANITY_AVAILABLE:
            result = profanity.censor(result)
        
        # Remove advanced academic words that are too complex
        words = result.split()
        cleaned_words = []
        
        for word in words:
            # Extract the base word (remove punctuation)
            base_word = re.sub(r'[^\w\s]', '', word)
            
            # Check if word is in advanced_words list
            if base_word.lower() in self.advanced_words:
                # Keep original word since it's likely from original text
                cleaned_words.append(word)
            else:
                cleaned_words.append(word)
        
        result = ' '.join(cleaned_words)
        return result
    
    def paraphrase(self, text, intensity=0.6):
        """
        Main paraphrasing method that applies multiple techniques.
        Preserves paragraph structure from input.
        
        Args:
            text: Input text to paraphrase
            intensity: Strength of paraphrasing (0.0 to 1.0)
        
        Returns:
            Paraphrased text with original paragraph structure preserved
        """
        if not text or not text.strip():
            return text
        
        # Split by paragraphs (double newline or single newline)
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            # Try splitting by single newlines
            paragraphs = text.split('\n')
        
        paraphrased_paragraphs = []
        
        # Process each paragraph separately
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                paraphrased_paragraphs.append('')
                continue
            
            # Apply techniques in sequence to each paragraph
            result = paragraph
            
            # Step 1: Replace with synonyms
            result = self.replace_with_synonyms(result, intensity)
            
            # Step 2: Add variations
            result = self.add_variations(result)
            
            # Step 3: Restructure
            if intensity > 0.5:
                result = self.restructure_sentences(result)
            
            # Step 4: Filter content for safety
            result = self.filter_content(result)
            
            paraphrased_paragraphs.append(result)
        
        # Rejoin paragraphs with double newlines
        return '\n\n'.join(paraphrased_paragraphs)


class SemanticValidator:
    """
    QA validator that checks if paraphrased text maintains semantic meaning
    while being appropriately humanized.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def extract_key_terms(self, text):
        """Extract key terms (nouns and important verbs) from text."""
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        key_terms = set()
        for word, pos in pos_tags:
            # Keep nouns, verbs, and adjectives
            if pos in ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ']:
                if word not in self.stop_words and word.isalpha() and len(word) > 2:
                    key_terms.add(word)
        
        return key_terms
    
    def calculate_semantic_similarity(self, original_text, paraphrased_text):
        """
        Calculate semantic similarity between original and paraphrased text.
        Returns a score from 0 to 100 indicating preservation of meaning.
        
        Args:
            original_text: Original input text
            paraphrased_text: Paraphrased output text
            
        Returns:
            Dictionary with similarity metrics and validation results
        """
        # Extract key terms from both texts
        original_terms = self.extract_key_terms(original_text)
        paraphrased_terms = self.extract_key_terms(paraphrased_text)
        
        # Calculate overlap
        common_terms = original_terms.intersection(paraphrased_terms)
        if len(original_terms) == 0:
            similarity_score = 100
        else:
            similarity_score = (len(common_terms) / len(original_terms)) * 100
        
        # Check for missing key concepts
        missing_terms = original_terms - paraphrased_terms
        added_terms = paraphrased_terms - original_terms
        
        # Calculate text length ratio (should be similar)
        original_length = len(original_text.split())
        paraphrased_length = len(paraphrased_text.split())
        length_ratio = min(paraphrased_length, original_length) / max(paraphrased_length, original_length) * 100
        
        # Check for negation preservation
        negation_preserved = self._check_negation_preservation(original_text, paraphrased_text)
        
        # Check for question format preservation
        question_preserved = self._check_question_preservation(original_text, paraphrased_text)
        
        # Check for proper noun preservation
        proper_nouns_preserved = self._check_proper_noun_preservation(original_text, paraphrased_text)
        
        # Check for number/date preservation
        numbers_dates_preserved = self._check_number_date_preservation(original_text, paraphrased_text)
        
        # Overall assessment - stricter criteria
        is_semantic_match = (
            similarity_score >= 70 and  # 70% threshold for acceptable paraphrase
            negation_preserved and
            question_preserved and
            proper_nouns_preserved and
            numbers_dates_preserved
        )
        is_humanized = paraphrased_length != original_length  # Should have some changes
        
        return {
            'similarity_score': round(similarity_score, 2),
            'length_similarity': round(length_ratio, 2),
            'original_key_terms': len(original_terms),
            'preserved_terms': len(common_terms),
            'missing_terms': list(missing_terms)[:5] if missing_terms else [],  # Show first 5
            'new_terms_added': len(added_terms),
            'semantic_match': is_semantic_match,
            'is_humanized': is_humanized,
            'negation_preserved': negation_preserved,
            'question_preserved': question_preserved,
            'proper_nouns_preserved': proper_nouns_preserved,
            'numbers_dates_preserved': numbers_dates_preserved,
            'quality_status': self._get_quality_status(similarity_score, is_humanized, negation_preserved, question_preserved),
            'recommendations': self._get_recommendations(similarity_score, is_humanized, missing_terms, negation_preserved, question_preserved)
        }
    
    def _check_negation_preservation(self, original_text, paraphrased_text):
        """Check if negations are preserved between original and paraphrased text."""
        negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 
                         'neither', 'nor', 'hardly', 'barely', 'scarcely', 'dont', 'doesnt', 
                         'didnt', 'wont', 'wouldnt', 'cant', 'couldnt', 'shouldnt', 'isnt', 
                         'arent', 'wasnt', 'werent', 'havent', 'hasnt', 'hadnt'}
        
        original_lower = original_text.lower()
        paraphrased_lower = paraphrased_text.lower()
        
        # Count negations in both texts
        original_negations = sum(1 for word in negation_words if word in original_lower)
        paraphrased_negations = sum(1 for word in negation_words if word in paraphrased_lower)
        
        # Negations should be preserved (same count or similar)
        return original_negations == paraphrased_negations
    
    def _check_question_preservation(self, original_text, paraphrased_text):
        """Check if question format is preserved."""
        original_is_question = '?' in original_text or any(
            original_text.strip().lower().startswith(q) 
            for q in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']
        )
        paraphrased_is_question = '?' in paraphrased_text or any(
            paraphrased_text.strip().lower().startswith(q) 
            for q in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']
        )
        
        return original_is_question == paraphrased_is_question
    
    def _check_proper_noun_preservation(self, original_text, paraphrased_text):
        """Check if proper nouns (capitalized words not at sentence start) are preserved."""
        import re
        
        # Extract proper nouns (capitalized words not at sentence start)
        original_proper = set(re.findall(r'\b[A-Z][a-z]+\b', original_text))
        paraphrased_proper = set(re.findall(r'\b[A-Z][a-z]+\b', paraphrased_text))
        
        # Check if most proper nouns are preserved
        if not original_proper:
            return True
        
        preserved = len(original_proper.intersection(paraphrased_proper))
        return preserved / len(original_proper) >= 0.7  # At least 70% preserved
    
    def _check_number_date_preservation(self, original_text, paraphrased_text):
        """Check if numbers and dates are preserved."""
        import re
        
        # Extract numbers
        original_numbers = set(re.findall(r'\b\d+\b', original_text))
        paraphrased_numbers = set(re.findall(r'\b\d+\b', paraphrased_text))
        
        # Check if numbers are preserved
        if not original_numbers:
            return True
        
        preserved = len(original_numbers.intersection(paraphrased_numbers))
        return preserved / len(original_numbers) >= 0.8  # At least 80% preserved
    
    def improve_paraphrase(self, original_text, paraphrased_text, engine):
        """
        Intelligently improve paraphrased text based on validation results.
        Works internally without user interaction.
        
        Args:
            original_text: Original input text
            paraphrased_text: Current paraphrased text
            engine: ParaphraserEngine instance to re-paraphrase if needed
            
        Returns:
            Improved paraphrased text
        """
        validation = self.calculate_semantic_similarity(original_text, paraphrased_text)
        
        # If semantic match is good (>=70%) and humanized, return as-is
        if validation['semantic_match'] and validation['is_humanized']:
            return paraphrased_text
        
        # If negation is lost, try to restore it
        if not validation['negation_preserved']:
            improved = self._restore_negation(original_text, paraphrased_text)
            # Re-validate after restoration
            new_validation = self.calculate_semantic_similarity(original_text, improved)
            if new_validation['negation_preserved']:
                paraphrased_text = improved
                validation = new_validation
        
        # If question format is lost, try to restore it
        if not validation['question_preserved']:
            improved = self._restore_question_format(original_text, paraphrased_text)
            # Re-validate after restoration
            new_validation = self.calculate_semantic_similarity(original_text, improved)
            if new_validation['question_preserved']:
                paraphrased_text = improved
                validation = new_validation
        
        # If similarity is too low, try to incorporate missing terms
        if validation['similarity_score'] < 70 and validation['missing_terms']:
            improved = self._reincorporate_missing_terms(
                paraphrased_text, 
                list(validation['missing_terms'])[:5]
            )
            # Re-validate after reincorporation
            new_validation = self.calculate_semantic_similarity(original_text, improved)
            if new_validation['similarity_score'] > validation['similarity_score']:
                paraphrased_text = improved
                validation = new_validation
        
        # If not humanized enough, apply more humanization
        if not validation['is_humanized']:
            # Text is too similar - needs more changes
            # Re-paraphrase with slightly higher internal intensity
            improved = engine.replace_with_synonyms(paraphrased_text, intensity=0.4)
            improved = engine.add_variations(improved)
            # Re-validate after humanization
            new_validation = self.calculate_semantic_similarity(original_text, improved)
            if new_validation['is_humanized']:
                paraphrased_text = improved
                validation = new_validation
        
        # Final safety check: if quality is still poor, return original text
        if validation['similarity_score'] < 50:
            # Too much meaning lost - return original
            return original_text
        
        return paraphrased_text
    
    def _restore_negation(self, original_text, paraphrased_text):
        """Try to restore negation from original text to paraphrased text."""
        negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'nowhere', 
                         'neither', 'nor', 'hardly', 'barely', 'scarcely'}
        
        original_lower = original_text.lower()
        paraphrased_lower = paraphrased_text.lower()
        
        # Find negations in original that are missing in paraphrased
        missing_negations = []
        for neg in negation_words:
            if neg in original_lower and neg not in paraphrased_lower:
                missing_negations.append(neg)
        
        if not missing_negations:
            return paraphrased_text
        
        # Try to add missing negations back
        result = paraphrased_text
        for neg in missing_negations[:1]:  # Only add one negation to avoid over-correction
            # Find a good place to insert the negation
            # Look for common patterns like "is", "are", "was", "were", "have", "has"
            import re
            pattern = r'\b(is|are|was|were|have|has|had|can|could|will|would|should|must)\b'
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            
            if matches:
                # Insert negation after the first match
                match = matches[0]
                insert_pos = match.end()
                result = result[:insert_pos] + ' ' + neg + result[insert_pos:]
                break
        
        return result
    
    def _restore_question_format(self, original_text, paraphrased_text):
        """Try to restore question format from original text to paraphrased text."""
        original_is_question = '?' in original_text or any(
            original_text.strip().lower().startswith(q) 
            for q in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']
        )
        
        if not original_is_question:
            return paraphrased_text
        
        # If original is a question, make sure paraphrased is too
        result = paraphrased_text
        
        # Add question mark if missing
        if '?' not in result:
            # Remove any trailing punctuation and add question mark
            result = result.rstrip('.!') + '?'
        
        # Ensure it starts with a question word if original did
        original_first_word = original_text.strip().split()[0].lower() if original_text.strip() else ''
        if original_first_word in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']:
            result_first_word = result.strip().split()[0].lower() if result.strip() else ''
            if result_first_word not in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']:
                # Try to restructure to start with question word
                # This is a simple approach - could be improved
                words = result.split()
                if len(words) > 1:
                    # Move the question word to the beginning if it exists
                    for i, word in enumerate(words):
                        if word.lower() in ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose', 'whom']:
                            words.insert(0, words.pop(i))
                            result = ' '.join(words)
                            break
        
        return result
    
    def _reincorporate_missing_terms(self, paraphrased_text, missing_terms):
        """
        Try to reincorporate important missing terms where they make sense.
        More context-aware to avoid breaking grammar or meaning.
        """
        result = paraphrased_text
        
        # Find sentences that might be missing the concepts
        sentences = sent_tokenize(paraphrased_text)
        modified_sentences = []
        
        for sentence in sentences:
            modified = sentence
            # Check each missing term
            for term in missing_terms:
                term_lower = term.lower()
                if term_lower not in modified.lower() and len(modified) > 30:
                    # This sentence could benefit from including this term
                    # Try to add it naturally by finding a good insertion point
                    
                    # Look for common patterns where we can insert the term
                    # Pattern 1: After "and", "or", "but", "also"
                    import re
                    connectors = r'\b(and|or|but|also|as well as|plus)\b'
                    matches = list(re.finditer(connectors, modified, re.IGNORECASE))
                    
                    if matches:
                        # Insert after the first connector
                        match = matches[0]
                        insert_pos = match.end()
                        # Add space if needed
                        if insert_pos < len(modified) and modified[insert_pos] != ' ':
                            modified = modified[:insert_pos] + ' ' + term + modified[insert_pos:]
                        else:
                            modified = modified[:insert_pos] + term + modified[insert_pos:]
                        break
                    
                    # Pattern 2: Before a comma or period
                    punctuation_matches = list(re.finditer(r'[,\.]', modified))
                    if punctuation_matches and len(punctuation_matches) > 1:
                        # Insert before the second-to-last punctuation
                        match = punctuation_matches[-2]
                        insert_pos = match.start()
                        # Add space before the term
                        modified = modified[:insert_pos] + ', ' + term + modified[insert_pos:]
                        break
                    
                    # Pattern 3: At the end of the sentence (before period)
                    if modified.endswith('.'):
                        modified = modified[:-1] + ', including ' + term + '.'
                        break
            
            modified_sentences.append(modified)
        
        return ' '.join(modified_sentences)
    
    def _get_quality_status(self, similarity_score, is_humanized, negation_preserved=True, question_preserved=True):
        """Determine quality status of paraphrase (internal only)."""
        if similarity_score < 60:
            return "POOR"
        elif similarity_score < 70:
            return "FAIR"
        elif not is_humanized:
            return "NOT_HUMANIZED"
        elif not negation_preserved:
            return "NEGATION_LOST"
        elif not question_preserved:
            return "QUESTION_LOST"
        else:
            return "GOOD"
    
    def _get_recommendations(self, similarity_score, is_humanized, missing_terms, negation_preserved=True, question_preserved=True):
        """Generate internal recommendations (not for user display)."""
        recommendations = []
        
        if similarity_score < 70:
            recommendations.append("low_similarity")
        
        if not is_humanized:
            recommendations.append("needs_humanization")
        
        if missing_terms:
            recommendations.append("missing_terms")
        
        if not negation_preserved:
            recommendations.append("negation_lost")
        
        if not question_preserved:
            recommendations.append("question_lost")
        
        return recommendations

