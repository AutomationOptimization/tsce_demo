# symbolic_transform_json

## Benchmark
- Train shells: 4
- Eval shells: 2

## Best Anchor
- Anchor key: `a3d3b83d7e54`
- Train pass rate: 0.25
- Best-layer alignment: 0.9627977308230605
- Mean vector stability: 0.8786085719863573

## Controls
- Shuffled train pass rate: 0.5
- Bad-anchor train pass rate: 0.25
- Context-collision eval pass rate: 0.5

## Selected Layers
- Layers: [41, 40]
- Layer summary: {
  "0": {
    "good_alignment": 0.9340845097068596,
    "shuffled_alignment": 0.9383412899285949,
    "bad_alignment": 0.9412398513323401,
    "control_alignment": 0.9412398513323401,
    "separation": -0.0071553416254804825,
    "mean_norm": 8.880998492240906,
    "good_vector_count": 8
  },
  "1": {
    "good_alignment": 0.9115811980085518,
    "shuffled_alignment": 0.9200687758233452,
    "bad_alignment": 0.9247338717960383,
    "control_alignment": 0.9247338717960383,
    "separation": -0.013152673787486524,
    "mean_norm": 7.282378613948822,
    "good_vector_count": 8
  },
  "2": {
    "good_alignment": 0.9014545256532225,
    "shuffled_alignment": 0.8974228042499351,
    "bad_alignment": 0.9091272338454357,
    "control_alignment": 0.9091272338454357,
    "separation": -0.007672708192213218,
    "mean_norm": 13.60211855173111,
    "good_vector_count": 8
  },
  "3": {
    "good_alignment": 0.8559935316321126,
    "shuffled_alignment": 0.857335129721369,
    "bad_alignment": 0.8697554924468225,
    "control_alignment": 0.8697554924468225,
    "separation": -0.013761960814709928,
    "mean_norm": 16.528811037540436,
    "good_vector_count": 8
  },
  "4": {
    "good_alignment": 0.8886753100995286,
    "shuffled_alignment": 0.8970040139491182,
    "bad_alignment": 0.8944060200688907,
    "control_alignment": 0.8970040139491182,
    "separation": -0.008328703849589614,
    "mean_norm": 15.918391644954681,
    "good_vector_count": 8
  },
  "5": {
    "good_alignment": 0.90615755848201,
    "shuffled_alignment": 0.9332210043299853,
    "bad_alignment": 0.9100515118887459,
    "control_alignment": 0.9332210043299853,
    "separation": -0.027063445847975376,
    "mean_norm": 20.623436212539673,
    "good_vector_count": 8
  },
  "6": {
    "good_alignment": 0.8435777148874394,
    "shuffled_alignment": 0.850906359364792,
    "bad_alignment": 0.8473401328047075,
    "control_alignment": 0.850906359364792,
    "separation": -0.007328644477352597,
    "mean_norm": 17.606569051742554,
    "good_vector_count": 8
  },
  "7": {
    "good_alignment": 0.8509416099441662,
    "shuffled_alignment": 0.8587355515254983,
    "bad_alignment": 0.8536949555936788,
    "control_alignment": 0.8587355515254983,
    "separation": -0.0077939415813320645,
    "mean_norm": 21.82322120666504,
    "good_vector_count": 8
  },
  "8": {
    "good_alignment": 0.8381010646692951,
    "shuffled_alignment": 0.8505900789865722,
    "bad_alignment": 0.8314477931882748,
    "control_alignment": 0.8505900789865722,
    "separation": -0.012489014317277114,
    "mean_norm": 10.93936139345169,
    "good_vector_count": 8
  },
  "9": {
    "good_alignment": 0.8340518061799196,
    "shuffled_alignment": 0.8415111372707339,
    "bad_alignment": 0.8235650374101503,
    "control_alignment": 0.8415111372707339,
    "separation": -0.007459331090814314,
    "mean_norm": 10.761331588029861,
    "good_vector_count": 8
  },
  "10": {
    "good_alignment": 0.9028615343317192,
    "shuffled_alignment": 0.8914437663329334,
    "bad_alignment": 0.8898758321043996,
    "control_alignment": 0.8914437663329334,
    "separation": 0.0114177679987858,
    "mean_norm": 14.059989094734192,
    "good_vector_count": 8
  },
  "11": {
    "good_alignment": 0.9319774432045568,
    "shuffled_alignment": 0.9588481383084466,
    "bad_alignment": 0.9565351727375138,
    "control_alignment": 0.9588481383084466,
    "separation": -0.02687069510388984,
    "mean_norm": 25.441166043281555,
    "good_vector_count": 8
  },
  "12": {
    "good_alignment": 0.9275412757604761,
    "shuffled_alignment": 0.9513384233211908,
    "bad_alignment": 0.9474561006282313,
    "control_alignment": 0.9513384233211908,
    "separation": -0.023797147560714715,
    "mean_norm": 27.026883363723755,
    "good_vector_count": 8
  },
  "13": {
    "good_alignment": 0.8931411538524061,
    "shuffled_alignment": 0.9166433489806214,
    "bad_alignment": 0.909291897597632,
    "control_alignment": 0.9166433489806214,
    "separation": -0.023502195128215275,
    "mean_norm": 28.612770318984985,
    "good_vector_count": 8
  },
  "14": {
    "good_alignment": 0.8810946854317127,
    "shuffled_alignment": 0.9002600778052113,
    "bad_alignment": 0.9036666298263584,
    "control_alignment": 0.9036666298263584,
    "separation": -0.022571944394645715,
    "mean_norm": 34.35810947418213,
    "good_vector_count": 8
  },
  "15": {
    "good_alignment": 0.8872978832552596,
    "shuffled_alignment": 0.895909381491406,
    "bad_alignment": 0.8896980582912491,
    "control_alignment": 0.895909381491406,
    "separation": -0.008611498236146375,
    "mean_norm": 33.875900864601135,
    "good_vector_count": 8
  },
  "16": {
    "good_alignment": 0.8609485940448777,
    "shuffled_alignment": 0.8587389392048027,
    "bad_alignment": 0.8587098419688292,
    "control_alignment": 0.8587389392048027,
    "separation": 0.0022096548400749594,
    "mean_norm": 37.74186944961548,
    "good_vector_count": 8
  },
  "17": {
    "good_alignment": 0.8301575619302181,
    "shuffled_alignment": 0.8232740814713644,
    "bad_alignment": 0.8331516955933632,
    "control_alignment": 0.8331516955933632,
    "separation": -0.00299413366314516,
    "mean_norm": 44.73918890953064,
    "good_vector_count": 8
  },
  "18": {
    "good_alignment": 0.8101039253922364,
    "shuffled_alignment": 0.7983870318631853,
    "bad_alignment": 0.8054605531758242,
    "control_alignment": 0.8054605531758242,
    "separation": 0.0046433722164122004,
    "mean_norm": 44.39546275138855,
    "good_vector_count": 8
  },
  "19": {
    "good_alignment": 0.7603737467368188,
    "shuffled_alignment": 0.7370582117624638,
    "bad_alignment": 0.7456898942679832,
    "control_alignment": 0.7456898942679832,
    "separation": 0.014683852468835634,
    "mean_norm": 44.12554430961609,
    "good_vector_count": 8
  },
  "20": {
    "good_alignment": 0.7625913729525423,
    "shuffled_alignment": 0.7491039858934719,
    "bad_alignment": 0.7439555146184914,
    "control_alignment": 0.7491039858934719,
    "separation": 0.013487387059070377,
    "mean_norm": 43.40270733833313,
    "good_vector_count": 8
  },
  "21": {
    "good_alignment": 0.7231909822811665,
    "shuffled_alignment": 0.7005414197869998,
    "bad_alignment": 0.7062121923452646,
    "control_alignment": 0.7062121923452646,
    "separation": 0.01697878993590196,
    "mean_norm": 44.67448675632477,
    "good_vector_count": 8
  },
  "22": {
    "good_alignment": 0.6792560029524269,
    "shuffled_alignment": 0.6351684893026966,
    "bad_alignment": 0.644857887030568,
    "control_alignment": 0.644857887030568,
    "separation": 0.03439811592185893,
    "mean_norm": 33.022695541381836,
    "good_vector_count": 8
  },
  "23": {
    "good_alignment": 0.6879124435467779,
    "shuffled_alignment": 0.647964573672625,
    "bad_alignment": 0.6613786665472946,
    "control_alignment": 0.6613786665472946,
    "separation": 0.02653377699948334,
    "mean_norm": 34.2170284986496,
    "good_vector_count": 8
  },
  "24": {
    "good_alignment": 0.6559116705848672,
    "shuffled_alignment": 0.6081692172328934,
    "bad_alignment": 0.619600510612909,
    "control_alignment": 0.619600510612909,
    "separation": 0.03631115997195822,
    "mean_norm": 33.45187783241272,
    "good_vector_count": 8
  },
  "25": {
    "good_alignment": 0.6198531188868982,
    "shuffled_alignment": 0.565078824672177,
    "bad_alignment": 0.5808901458579359,
    "control_alignment": 0.5808901458579359,
    "separation": 0.038962973028962344,
    "mean_norm": 35.33837151527405,
    "good_vector_count": 8
  },
  "26": {
    "good_alignment": 0.5950096026541087,
    "shuffled_alignment": 0.5331065051871545,
    "bad_alignment": 0.5501806029599133,
    "control_alignment": 0.5501806029599133,
    "separation": 0.044828999694195404,
    "mean_norm": 34.62922382354736,
    "good_vector_count": 8
  },
  "27": {
    "good_alignment": 0.5760858888647217,
    "shuffled_alignment": 0.5098944496444453,
    "bad_alignment": 0.5357435393138889,
    "control_alignment": 0.5357435393138889,
    "separation": 0.04034234955083282,
    "mean_norm": 34.26413655281067,
    "good_vector_count": 8
  },
  "28": {
    "good_alignment": 0.5852299548933081,
    "shuffled_alignment": 0.5194671475771593,
    "bad_alignment": 0.5419397711046068,
    "control_alignment": 0.5419397711046068,
    "separation": 0.04329018378870131,
    "mean_norm": 30.474790334701538,
    "good_vector_count": 8
  },
  "29": {
    "good_alignment": 0.6084277383136708,
    "shuffled_alignment": 0.5630216843489465,
    "bad_alignment": 0.5712788581072811,
    "control_alignment": 0.5712788581072811,
    "separation": 0.037148880206389734,
    "mean_norm": 30.414462327957153,
    "good_vector_count": 8
  },
  "30": {
    "good_alignment": 0.5923162446300047,
    "shuffled_alignment": 0.5282960239996275,
    "bad_alignment": 0.5459081268764772,
    "control_alignment": 0.5459081268764772,
    "separation": 0.04640811775352749,
    "mean_norm": 29.280154585838318,
    "good_vector_count": 8
  },
  "31": {
    "good_alignment": 0.5809140498432194,
    "shuffled_alignment": 0.5059682458518387,
    "bad_alignment": 0.5316384973901456,
    "control_alignment": 0.5316384973901456,
    "separation": 0.0492755524530738,
    "mean_norm": 30.537063479423523,
    "good_vector_count": 8
  },
  "32": {
    "good_alignment": 0.5690748476968127,
    "shuffled_alignment": 0.4908088455317199,
    "bad_alignment": 0.5318973914640729,
    "control_alignment": 0.5318973914640729,
    "separation": 0.037177456232739825,
    "mean_norm": 31.812856912612915,
    "good_vector_count": 8
  },
  "33": {
    "good_alignment": 0.5732805229231213,
    "shuffled_alignment": 0.4841075952972573,
    "bad_alignment": 0.5225977609248313,
    "control_alignment": 0.5225977609248313,
    "separation": 0.050682761998289916,
    "mean_norm": 32.72562110424042,
    "good_vector_count": 8
  },
  "34": {
    "good_alignment": 0.5473934921620353,
    "shuffled_alignment": 0.47313284960951135,
    "bad_alignment": 0.49478514596592227,
    "control_alignment": 0.49478514596592227,
    "separation": 0.05260834619611299,
    "mean_norm": 33.65519738197327,
    "good_vector_count": 8
  },
  "35": {
    "good_alignment": 0.5593726731561812,
    "shuffled_alignment": 0.48924625023946794,
    "bad_alignment": 0.5042275379423947,
    "control_alignment": 0.5042275379423947,
    "separation": 0.0551451352137865,
    "mean_norm": 37.742045879364014,
    "good_vector_count": 8
  },
  "36": {
    "good_alignment": 0.5428921753999651,
    "shuffled_alignment": 0.46722328166558635,
    "bad_alignment": 0.4826302301435769,
    "control_alignment": 0.4826302301435769,
    "separation": 0.06026194525638823,
    "mean_norm": 40.2281711101532,
    "good_vector_count": 8
  },
  "37": {
    "good_alignment": 0.5319667532012936,
    "shuffled_alignment": 0.4454823435640351,
    "bad_alignment": 0.459370983324394,
    "control_alignment": 0.459370983324394,
    "separation": 0.07259576987689959,
    "mean_norm": 43.98980522155762,
    "good_vector_count": 8
  },
  "38": {
    "good_alignment": 0.4945996928943572,
    "shuffled_alignment": 0.4093197380003602,
    "bad_alignment": 0.42455538686913935,
    "control_alignment": 0.42455538686913935,
    "separation": 0.07004430602521783,
    "mean_norm": 46.72373032569885,
    "good_vector_count": 8
  },
  "39": {
    "good_alignment": 0.4549863501530648,
    "shuffled_alignment": 0.3674002568218868,
    "bad_alignment": 0.3769585409922844,
    "control_alignment": 0.3769585409922844,
    "separation": 0.07802780916078045,
    "mean_norm": 49.92699337005615,
    "good_vector_count": 8
  },
  "40": {
    "good_alignment": 0.4248437420975755,
    "shuffled_alignment": 0.3382399859469873,
    "bad_alignment": 0.34097789271956663,
    "control_alignment": 0.34097789271956663,
    "separation": 0.0838658493780089,
    "mean_norm": 48.261613607406616,
    "good_vector_count": 8
  },
  "41": {
    "good_alignment": 0.39864510657763097,
    "shuffled_alignment": 0.30693509319892104,
    "bad_alignment": 0.3100389722639639,
    "control_alignment": 0.3100389722639639,
    "separation": 0.0886061343136671,
    "mean_norm": 143.52159309387207,
    "good_vector_count": 8
  }
}

## Invariants
- Order sensitivity: 0.0
- Core tokens: ['scope', 'about', 'above', 'while', 'since', 'meets', 'ought', 'value', 'going', 'reach', 'focus', 'these', 'thing', 'learn', 'plus', 'threey', 'twelve', 'eighty', 'modul', 'aspect', 'detail', 'power', 'rule', 'nabla', 'grade', 'shift', 'obtain', 'yields', 'ideal', 'jolly', 'knife', 'lemon', 'noble', 'ocean', 'quest', 'reason', 'taste', 'umbra', 'vivid', 'wagon', 'yacht', 'input', 'tours', 'twice', 'curve', 'stream', 'theory', 'begin', 'level', 'user', 'wants', 'need', 'output', 'somehow', 'encode', 'process']
- Proxy tokens: []
- Padding tokens: ['given', 'count', 'solve', 'basis', 'logic', 'sense', 'world', 'form', 'prompt', 'query', 'whereby', 'delta', 'cubic', 'factor', 'prime', 'exact', 'derive', 'minus', 'change', 'degree', 'slopey', 'versus', 'hencey', 'leasty', 'must', 'diverse', 'reuse', 'relate', 'final', 'check', 'alpha', 'berry', 'doodle', 'event', 'flame', 'graph', 'constant', 'multiple', 'combine', 'salsa', 'plane', 'fuzzy', 'axiom', 'below', 'inner', 'outer', 'comma', 'lbrace', 'omega', 'zeta', 'iota', 'kappa', 'which', 'method']

## Causal
- Summary: {
  "context_collision_anchor": {
    "count": 2,
    "pass_rate": 0.5,
    "mean_score": 0.5,
    "mean_latency_s": 9.499786285567097,
    "mean_logit_cosine_shift": 0.045147357972259355,
    "mean_attention_to_control": null
  },
  "best_anchor_eval": {
    "count": 2,
    "pass_rate": 1.0,
    "mean_score": 1.0,
    "mean_latency_s": 9.581617428106256,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_class_delta": {
    "count": 2,
    "pass_rate": 0.5,
    "mean_score": 0.5,
    "mean_latency_s": 1.3820228449767455,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_bad_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 1.4151611641282216,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_shuffled_delta": {
    "count": 2,
    "pass_rate": 0.5,
    "mean_score": 0.5,
    "mean_latency_s": 16.926798665896058,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "remove_class_delta": {
    "count": 2,
    "pass_rate": 1.0,
    "mean_score": 1.0,
    "mean_latency_s": 9.291689195553772,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "project_out_class_delta": {
    "count": 2,
    "pass_rate": 0.5,
    "mean_score": 0.5,
    "mean_latency_s": 1.3366139675490558,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  }
}

## Compression / Outlier
- Summary: {
  "good": {
    "layer_norms": {
      "41": 143.52159309387207,
      "40": 48.261613607406616
    },
    "norm_concentration": 0.7483532868309014,
    "stability": 0.6255954280495644,
    "top_abs_mass_ratio": 0.04739146679639816
  },
  "bad": {
    "layer_norms": {
      "41": 161.25211906433105,
      "40": 53.320374488830566
    },
    "norm_concentration": 0.7515041485239574,
    "stability": 0.6400014087557793,
    "top_abs_mass_ratio": 0.04336369410157204
  },
  "shuffled": {
    "layer_norms": {
      "41": 135.1600170135498,
      "40": 45.802048206329346
    },
    "norm_concentration": 0.7468969634565273,
    "stability": 0.6333445012569427,
    "top_abs_mass_ratio": 0.048233628273010254
  },
  "good_minus_controls": {
    "norm_concentration_delta": -0.0031508616930560063,
    "stability_delta": -0.014405980706214905,
    "top_abs_mass_ratio_delta": -0.0008421614766120911
  }
}

## Bad Anchors
- `af39cf22c102`: pass_rate=0.25 alignment=0.9565351727375138
