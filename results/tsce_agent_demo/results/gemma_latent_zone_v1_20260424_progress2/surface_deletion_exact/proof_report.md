# surface_deletion_exact

## Benchmark
- Train shells: 4
- Eval shells: 2

## Best Anchor
- Anchor key: `be80ce2d73d2`
- Train pass rate: 0.5
- Best-layer alignment: 0.5021669055621848
- Mean vector stability: 0.6408155114345607

## Controls
- Shuffled train pass rate: 0.25
- Bad-anchor train pass rate: 0.25
- Context-collision eval pass rate: 0.0

## Selected Layers
- Layers: [0, 2]
- Layer summary: {
  "0": {
    "good_alignment": 0.44904396776639743,
    "shuffled_alignment": 0.27484108655317235,
    "bad_alignment": 0.2780362372345637,
    "control_alignment": 0.2780362372345637,
    "separation": 0.1710077305318337,
    "mean_norm": 34.61085882782936,
    "good_vector_count": 8
  },
  "1": {
    "good_alignment": 0.4733976808344935,
    "shuffled_alignment": 0.34936528279052254,
    "bad_alignment": 0.3764700010707915,
    "control_alignment": 0.3764700010707915,
    "separation": 0.09692767976370203,
    "mean_norm": 18.71142002940178,
    "good_vector_count": 8
  },
  "2": {
    "good_alignment": 0.5310087185740858,
    "shuffled_alignment": 0.38996904433740176,
    "bad_alignment": 0.4038819822449565,
    "control_alignment": 0.4038819822449565,
    "separation": 0.12712673632912935,
    "mean_norm": 37.60216528177261,
    "good_vector_count": 8
  },
  "3": {
    "good_alignment": 0.4487151368881869,
    "shuffled_alignment": 0.3391565623659288,
    "bad_alignment": 0.3186793156133658,
    "control_alignment": 0.3391565623659288,
    "separation": 0.10955857452225815,
    "mean_norm": 46.97759783267975,
    "good_vector_count": 8
  },
  "4": {
    "good_alignment": 0.46734152879791807,
    "shuffled_alignment": 0.34747697693906526,
    "bad_alignment": 0.3382197015616903,
    "control_alignment": 0.34747697693906526,
    "separation": 0.11986455185885281,
    "mean_norm": 43.80686551332474,
    "good_vector_count": 8
  },
  "5": {
    "good_alignment": 0.5326911139147584,
    "shuffled_alignment": 0.4292210396961542,
    "bad_alignment": 0.4079122857521451,
    "control_alignment": 0.4292210396961542,
    "separation": 0.10347007421860421,
    "mean_norm": 55.844731092453,
    "good_vector_count": 8
  },
  "6": {
    "good_alignment": 0.4817800995793587,
    "shuffled_alignment": 0.38698022064168125,
    "bad_alignment": 0.3809946033568076,
    "control_alignment": 0.38698022064168125,
    "separation": 0.09479987893767744,
    "mean_norm": 46.835473239421844,
    "good_vector_count": 8
  },
  "7": {
    "good_alignment": 0.4261171633408196,
    "shuffled_alignment": 0.34778251923229314,
    "bad_alignment": 0.33504689821511713,
    "control_alignment": 0.34778251923229314,
    "separation": 0.07833464410852647,
    "mean_norm": 59.9839471578598,
    "good_vector_count": 8
  },
  "8": {
    "good_alignment": 0.37812788189110813,
    "shuffled_alignment": 0.29958646484496326,
    "bad_alignment": 0.29218145459502254,
    "control_alignment": 0.29958646484496326,
    "separation": 0.07854141704614487,
    "mean_norm": 44.8982617855072,
    "good_vector_count": 8
  },
  "9": {
    "good_alignment": 0.37454073877073096,
    "shuffled_alignment": 0.2886411419090356,
    "bad_alignment": 0.27264037202930613,
    "control_alignment": 0.2886411419090356,
    "separation": 0.08589959686169535,
    "mean_norm": 51.36720988154411,
    "good_vector_count": 8
  },
  "10": {
    "good_alignment": 0.4230692134323505,
    "shuffled_alignment": 0.31436603536275637,
    "bad_alignment": 0.30413752764911267,
    "control_alignment": 0.31436603536275637,
    "separation": 0.10870317806959412,
    "mean_norm": 60.07275879383087,
    "good_vector_count": 8
  },
  "11": {
    "good_alignment": 0.5356073437949734,
    "shuffled_alignment": 0.41277962879461394,
    "bad_alignment": 0.41281065513789916,
    "control_alignment": 0.41281065513789916,
    "separation": 0.12279668865707427,
    "mean_norm": 61.14603638648987,
    "good_vector_count": 8
  },
  "12": {
    "good_alignment": 0.5515361905206775,
    "shuffled_alignment": 0.4280420548371519,
    "bad_alignment": 0.42319430588452045,
    "control_alignment": 0.4280420548371519,
    "separation": 0.12349413568352557,
    "mean_norm": 58.55528688430786,
    "good_vector_count": 8
  },
  "13": {
    "good_alignment": 0.5403581973555339,
    "shuffled_alignment": 0.4273036831174128,
    "bad_alignment": 0.41716381362324323,
    "control_alignment": 0.4273036831174128,
    "separation": 0.11305451423812113,
    "mean_norm": 59.54634988307953,
    "good_vector_count": 8
  },
  "14": {
    "good_alignment": 0.5389011951240766,
    "shuffled_alignment": 0.44304512882226926,
    "bad_alignment": 0.4321576428392245,
    "control_alignment": 0.44304512882226926,
    "separation": 0.09585606630180737,
    "mean_norm": 66.46573376655579,
    "good_vector_count": 8
  },
  "15": {
    "good_alignment": 0.5260564367261682,
    "shuffled_alignment": 0.4481822458291225,
    "bad_alignment": 0.4419224971182902,
    "control_alignment": 0.4481822458291225,
    "separation": 0.07787419089704567,
    "mean_norm": 68.69414401054382,
    "good_vector_count": 8
  },
  "16": {
    "good_alignment": 0.47279245786726654,
    "shuffled_alignment": 0.3965773258940879,
    "bad_alignment": 0.4072579602726294,
    "control_alignment": 0.4072579602726294,
    "separation": 0.06553449759463714,
    "mean_norm": 79.09990549087524,
    "good_vector_count": 8
  },
  "17": {
    "good_alignment": 0.4354688002604763,
    "shuffled_alignment": 0.36196014176979313,
    "bad_alignment": 0.3760721221499543,
    "control_alignment": 0.3760721221499543,
    "separation": 0.059396678110522017,
    "mean_norm": 88.117924451828,
    "good_vector_count": 8
  },
  "18": {
    "good_alignment": 0.4122702121478853,
    "shuffled_alignment": 0.33993055760082563,
    "bad_alignment": 0.34558322993684437,
    "control_alignment": 0.34558322993684437,
    "separation": 0.06668698221104091,
    "mean_norm": 90.27040791511536,
    "good_vector_count": 8
  },
  "19": {
    "good_alignment": 0.36429016661415253,
    "shuffled_alignment": 0.30413458070533134,
    "bad_alignment": 0.29870295088096943,
    "control_alignment": 0.30413458070533134,
    "separation": 0.060155585908821196,
    "mean_norm": 92.50004196166992,
    "good_vector_count": 8
  },
  "20": {
    "good_alignment": 0.3488228002691118,
    "shuffled_alignment": 0.28828230910559255,
    "bad_alignment": 0.2926541260864189,
    "control_alignment": 0.2926541260864189,
    "separation": 0.05616867418269289,
    "mean_norm": 96.9409441947937,
    "good_vector_count": 8
  },
  "21": {
    "good_alignment": 0.3188500516057413,
    "shuffled_alignment": 0.2689463722536026,
    "bad_alignment": 0.2797883746694811,
    "control_alignment": 0.2797883746694811,
    "separation": 0.039061676936260226,
    "mean_norm": 101.26954102516174,
    "good_vector_count": 8
  },
  "22": {
    "good_alignment": 0.3088032088597875,
    "shuffled_alignment": 0.23784712713818243,
    "bad_alignment": 0.24558191731654877,
    "control_alignment": 0.24558191731654877,
    "separation": 0.06322129154323874,
    "mean_norm": 69.43551087379456,
    "good_vector_count": 8
  },
  "23": {
    "good_alignment": 0.36881536251262614,
    "shuffled_alignment": 0.2683140218284263,
    "bad_alignment": 0.28104574287930967,
    "control_alignment": 0.28104574287930967,
    "separation": 0.08776961963331648,
    "mean_norm": 73.72229743003845,
    "good_vector_count": 8
  },
  "24": {
    "good_alignment": 0.36426857330936324,
    "shuffled_alignment": 0.2494747883683782,
    "bad_alignment": 0.2622780194556018,
    "control_alignment": 0.2622780194556018,
    "separation": 0.10199055385376143,
    "mean_norm": 72.19358897209167,
    "good_vector_count": 8
  },
  "25": {
    "good_alignment": 0.35461413316872686,
    "shuffled_alignment": 0.24054626716875294,
    "bad_alignment": 0.2498890534714073,
    "control_alignment": 0.2498890534714073,
    "separation": 0.10472507969731956,
    "mean_norm": 74.1733729839325,
    "good_vector_count": 8
  },
  "26": {
    "good_alignment": 0.3302882280499366,
    "shuffled_alignment": 0.2193736151525962,
    "bad_alignment": 0.2332327837148259,
    "control_alignment": 0.2332327837148259,
    "separation": 0.09705544433511074,
    "mean_norm": 75.55708920955658,
    "good_vector_count": 8
  },
  "27": {
    "good_alignment": 0.324320416549316,
    "shuffled_alignment": 0.2180957365460859,
    "bad_alignment": 0.24009157035929532,
    "control_alignment": 0.24009157035929532,
    "separation": 0.08422884619002066,
    "mean_norm": 80.50248312950134,
    "good_vector_count": 8
  },
  "28": {
    "good_alignment": 0.33734659290435864,
    "shuffled_alignment": 0.21731797671242348,
    "bad_alignment": 0.2315577226312727,
    "control_alignment": 0.2315577226312727,
    "separation": 0.10578887027308595,
    "mean_norm": 80.96952092647552,
    "good_vector_count": 8
  },
  "29": {
    "good_alignment": 0.3730798610004692,
    "shuffled_alignment": 0.25097989966695317,
    "bad_alignment": 0.26236708030586525,
    "control_alignment": 0.26236708030586525,
    "separation": 0.11071278069460394,
    "mean_norm": 83.30248510837555,
    "good_vector_count": 8
  },
  "30": {
    "good_alignment": 0.3610101316922186,
    "shuffled_alignment": 0.23715780749050852,
    "bad_alignment": 0.2528054322621365,
    "control_alignment": 0.2528054322621365,
    "separation": 0.10820469943008215,
    "mean_norm": 86.60066771507263,
    "good_vector_count": 8
  },
  "31": {
    "good_alignment": 0.35825930360272296,
    "shuffled_alignment": 0.2402478622014627,
    "bad_alignment": 0.24848998503058675,
    "control_alignment": 0.24848998503058675,
    "separation": 0.1097693185721362,
    "mean_norm": 88.79150199890137,
    "good_vector_count": 8
  },
  "32": {
    "good_alignment": 0.3648332003437432,
    "shuffled_alignment": 0.24647108927681835,
    "bad_alignment": 0.25800002136553996,
    "control_alignment": 0.25800002136553996,
    "separation": 0.10683317897820327,
    "mean_norm": 89.74568521976471,
    "good_vector_count": 8
  },
  "33": {
    "good_alignment": 0.35191098974635937,
    "shuffled_alignment": 0.243888631791923,
    "bad_alignment": 0.24202597737395246,
    "control_alignment": 0.243888631791923,
    "separation": 0.10802235795443638,
    "mean_norm": 90.11342477798462,
    "good_vector_count": 8
  },
  "34": {
    "good_alignment": 0.3432074140297377,
    "shuffled_alignment": 0.24045891465250882,
    "bad_alignment": 0.23750238906719878,
    "control_alignment": 0.24045891465250882,
    "separation": 0.10274849937722888,
    "mean_norm": 90.31535685062408,
    "good_vector_count": 8
  },
  "35": {
    "good_alignment": 0.3432303649836456,
    "shuffled_alignment": 0.24315554914945517,
    "bad_alignment": 0.2516799516100232,
    "control_alignment": 0.2516799516100232,
    "separation": 0.09155041337362241,
    "mean_norm": 96.10582256317139,
    "good_vector_count": 8
  },
  "36": {
    "good_alignment": 0.3449339039382973,
    "shuffled_alignment": 0.24153901642464656,
    "bad_alignment": 0.25393372231263917,
    "control_alignment": 0.25393372231263917,
    "separation": 0.09100018162565815,
    "mean_norm": 97.36844491958618,
    "good_vector_count": 8
  },
  "37": {
    "good_alignment": 0.3281066065129915,
    "shuffled_alignment": 0.22004337059034515,
    "bad_alignment": 0.22547050857255815,
    "control_alignment": 0.22547050857255815,
    "separation": 0.10263609794043335,
    "mean_norm": 100.29594874382019,
    "good_vector_count": 8
  },
  "38": {
    "good_alignment": 0.3178894699357094,
    "shuffled_alignment": 0.21210504726441184,
    "bad_alignment": 0.21937364340386153,
    "control_alignment": 0.21937364340386153,
    "separation": 0.09851582653184784,
    "mean_norm": 102.08304023742676,
    "good_vector_count": 8
  },
  "39": {
    "good_alignment": 0.3038857845856948,
    "shuffled_alignment": 0.19707651549261432,
    "bad_alignment": 0.2022521173779536,
    "control_alignment": 0.2022521173779536,
    "separation": 0.10163366720774122,
    "mean_norm": 103.09895873069763,
    "good_vector_count": 8
  },
  "40": {
    "good_alignment": 0.2714302465751385,
    "shuffled_alignment": 0.1697281272401766,
    "bad_alignment": 0.16826463103056977,
    "control_alignment": 0.1697281272401766,
    "separation": 0.10170211933496187,
    "mean_norm": 96.99100279808044,
    "good_vector_count": 8
  },
  "41": {
    "good_alignment": 0.2715256296369536,
    "shuffled_alignment": 0.15805392833279788,
    "bad_alignment": 0.15385551237156356,
    "control_alignment": 0.15805392833279788,
    "separation": 0.11347170130415571,
    "mean_norm": 284.0505495071411,
    "good_vector_count": 8
  }
}

## Invariants
- Order sensitivity: 0.0
- Core tokens: ['double', 'hyphens', 'input', 'keeping', 'words', 'exactly', 'they', 'format', 'result', 'aweee', 'scope', 'logic', 'query', 'datum']
- Proxy tokens: []
- Padding tokens: ['realm', 'pulse', 'focus', 'shift', 'yield', 'plane', 'tribe', 'weave', 'honor', 'quest', 'gazed', 'sight', 'fleet', 'guard', 'hopes', 'limit', 'senserar', 'must', 'output', 'object', 'adheres', 'strict', 'token', 'ensuring', 'overlap', 'utilized', 'user', 'wants', 'dashes', 'sample', 'sentence', 'specific', 'letter', 'direct', 'reuse', 'banned', 'which', 'modified', 'prompt', 'asks', 'lines', 'apply', 'will', 'required', 'present', 'entirety', 'across', 'these', 'generate', 'given', 'conflict', 'between', 'content', 'fixed', 'complex', 'count', 'separate', 'among', 'where', 'about', 'maybe', 'above', 'below', 'though', 'since', 'would', 'small', 'issue', 'other', 'never', 'within', 'unless', 'always', 'already', 'diverse', 'ideas', 'usage', 'stream', 'change', 'alter', 'splice', 'merge', 'paths', 'gains', 'shows', 'seems', 'basis', 'solve', 'need', 'involved', 'response', 'abstract', 'task', 'actually', 'against', 'related']

## Causal
- Summary: {
  "context_collision_anchor": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 9.558718506945297,
    "mean_logit_cosine_shift": 0.013720132578406392,
    "mean_attention_to_control": null
  },
  "best_anchor_eval": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 6.356876257923432,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_class_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 9.242096794419922,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_bad_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 9.445063343038782,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_shuffled_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 9.474592672893777,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "remove_class_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 6.558484891895205,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "project_out_class_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 11.08808992349077,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  }
}

## Compression / Outlier
- Summary: {
  "good": {
    "layer_norms": {
      "0": 34.61085882782936,
      "2": 37.60216528177261
    },
    "norm_concentration": 0.5207116824895961,
    "stability": 0.44786067120730877,
    "top_abs_mass_ratio": 0.11349779739975929
  },
  "bad": {
    "layer_norms": {
      "0": 44.8859099149704,
      "2": 47.16289484500885
    },
    "norm_concentration": 0.5123683568514321,
    "stability": 0.4955262476578355,
    "top_abs_mass_ratio": 0.11168338730931282
  },
  "shuffled": {
    "layer_norms": {
      "0": 44.36014246940613,
      "2": 46.34127116203308
    },
    "norm_concentration": 0.5109211566463405,
    "stability": 0.4876620713621378,
    "top_abs_mass_ratio": 0.11174577474594116
  },
  "good_minus_controls": {
    "norm_concentration_delta": 0.008343325638163979,
    "stability_delta": -0.047665576450526714,
    "top_abs_mass_ratio_delta": 0.0017520226538181305
  }
}

## Bad Anchors
- `4f08ef51d353`: pass_rate=0.25 alignment=0.4419224971182902
