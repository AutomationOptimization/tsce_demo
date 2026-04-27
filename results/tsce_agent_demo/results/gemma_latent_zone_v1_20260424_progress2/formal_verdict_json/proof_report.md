# formal_verdict_json

## Benchmark
- Train shells: 4
- Eval shells: 2

## Best Anchor
- Anchor key: `082e7b7867d0`
- Train pass rate: 0.5
- Best-layer alignment: 0.9442820169392765
- Mean vector stability: 0.8350646382286435

## Controls
- Shuffled train pass rate: 0.5
- Bad-anchor train pass rate: 0.5
- Context-collision eval pass rate: 0.0

## Selected Layers
- Layers: [22, 27]
- Layer summary: {
  "0": {
    "good_alignment": 0.9058333619508012,
    "shuffled_alignment": 0.9024716470199752,
    "bad_alignment": 0.8945480771901014,
    "control_alignment": 0.9024716470199752,
    "separation": 0.003361714930826043,
    "mean_norm": 8.243245869874954,
    "good_vector_count": 8
  },
  "1": {
    "good_alignment": 0.9082912426592565,
    "shuffled_alignment": 0.9138147366102346,
    "bad_alignment": 0.9058493980050052,
    "control_alignment": 0.9138147366102346,
    "separation": -0.0055234939509780645,
    "mean_norm": 6.7386365830898285,
    "good_vector_count": 8
  },
  "2": {
    "good_alignment": 0.9317473033773301,
    "shuffled_alignment": 0.9299627298035967,
    "bad_alignment": 0.9357398540234937,
    "control_alignment": 0.9357398540234937,
    "separation": -0.003992550646163551,
    "mean_norm": 13.02945750951767,
    "good_vector_count": 8
  },
  "3": {
    "good_alignment": 0.8354944078891563,
    "shuffled_alignment": 0.8445268864713597,
    "bad_alignment": 0.8304708611586182,
    "control_alignment": 0.8445268864713597,
    "separation": -0.009032478582203352,
    "mean_norm": 17.706526458263397,
    "good_vector_count": 8
  },
  "4": {
    "good_alignment": 0.8826328199209306,
    "shuffled_alignment": 0.8922664687138872,
    "bad_alignment": 0.8807188081238045,
    "control_alignment": 0.8922664687138872,
    "separation": -0.009633648792956584,
    "mean_norm": 17.067691445350647,
    "good_vector_count": 8
  },
  "5": {
    "good_alignment": 0.8844714207430476,
    "shuffled_alignment": 0.9135217100054486,
    "bad_alignment": 0.891597963185665,
    "control_alignment": 0.9135217100054486,
    "separation": -0.02905028926240094,
    "mean_norm": 22.97106659412384,
    "good_vector_count": 8
  },
  "6": {
    "good_alignment": 0.8028577183067694,
    "shuffled_alignment": 0.8242891907852941,
    "bad_alignment": 0.7924567099387296,
    "control_alignment": 0.8242891907852941,
    "separation": -0.021431472478524616,
    "mean_norm": 20.125216245651245,
    "good_vector_count": 8
  },
  "7": {
    "good_alignment": 0.7275417732383408,
    "shuffled_alignment": 0.7444545410674889,
    "bad_alignment": 0.6831052699837552,
    "control_alignment": 0.7444545410674889,
    "separation": -0.01691276782914808,
    "mean_norm": 22.98488736152649,
    "good_vector_count": 8
  },
  "8": {
    "good_alignment": 0.7138321654010095,
    "shuffled_alignment": 0.7027124081094506,
    "bad_alignment": 0.6768559609499585,
    "control_alignment": 0.7027124081094506,
    "separation": 0.011119757291558896,
    "mean_norm": 11.461303949356079,
    "good_vector_count": 8
  },
  "9": {
    "good_alignment": 0.7452096129306262,
    "shuffled_alignment": 0.7381098936405456,
    "bad_alignment": 0.725829175741478,
    "control_alignment": 0.7381098936405456,
    "separation": 0.007099719290080575,
    "mean_norm": 11.249149858951569,
    "good_vector_count": 8
  },
  "10": {
    "good_alignment": 0.8450952180225861,
    "shuffled_alignment": 0.8456097875998099,
    "bad_alignment": 0.8340053203933269,
    "control_alignment": 0.8456097875998099,
    "separation": -0.0005145695772237335,
    "mean_norm": 14.061928629875183,
    "good_vector_count": 8
  },
  "11": {
    "good_alignment": 0.9190135307439716,
    "shuffled_alignment": 0.9452778567232695,
    "bad_alignment": 0.9359368319775329,
    "control_alignment": 0.9452778567232695,
    "separation": -0.026264325979297842,
    "mean_norm": 25.443829774856567,
    "good_vector_count": 8
  },
  "12": {
    "good_alignment": 0.9137434753043877,
    "shuffled_alignment": 0.9355136325949013,
    "bad_alignment": 0.9276728604944097,
    "control_alignment": 0.9355136325949013,
    "separation": -0.02177015729051357,
    "mean_norm": 27.062703609466553,
    "good_vector_count": 8
  },
  "13": {
    "good_alignment": 0.8657815006705148,
    "shuffled_alignment": 0.8888906063552368,
    "bad_alignment": 0.8820951233664834,
    "control_alignment": 0.8888906063552368,
    "separation": -0.02310910568472191,
    "mean_norm": 28.893230676651,
    "good_vector_count": 8
  },
  "14": {
    "good_alignment": 0.8527063981912975,
    "shuffled_alignment": 0.8739859236025614,
    "bad_alignment": 0.8622707156362588,
    "control_alignment": 0.8739859236025614,
    "separation": -0.02127952541126399,
    "mean_norm": 34.056426882743835,
    "good_vector_count": 8
  },
  "15": {
    "good_alignment": 0.8387319444452092,
    "shuffled_alignment": 0.8575833044082907,
    "bad_alignment": 0.8471401371029922,
    "control_alignment": 0.8575833044082907,
    "separation": -0.018851359963081515,
    "mean_norm": 33.68628716468811,
    "good_vector_count": 8
  },
  "16": {
    "good_alignment": 0.8082413543299563,
    "shuffled_alignment": 0.8261240080978745,
    "bad_alignment": 0.819858984149465,
    "control_alignment": 0.8261240080978745,
    "separation": -0.0178826537679182,
    "mean_norm": 37.911872148513794,
    "good_vector_count": 8
  },
  "17": {
    "good_alignment": 0.7463235071266566,
    "shuffled_alignment": 0.7681194453435772,
    "bad_alignment": 0.7602974511431655,
    "control_alignment": 0.7681194453435772,
    "separation": -0.021795938216920607,
    "mean_norm": 49.377057790756226,
    "good_vector_count": 8
  },
  "18": {
    "good_alignment": 0.7262565713802208,
    "shuffled_alignment": 0.7395881045040876,
    "bad_alignment": 0.7298663100267525,
    "control_alignment": 0.7395881045040876,
    "separation": -0.013331533123866812,
    "mean_norm": 48.31454348564148,
    "good_vector_count": 8
  },
  "19": {
    "good_alignment": 0.6940026875333812,
    "shuffled_alignment": 0.6995252988157622,
    "bad_alignment": 0.68898852789515,
    "control_alignment": 0.6995252988157622,
    "separation": -0.0055226112823809625,
    "mean_norm": 50.08715844154358,
    "good_vector_count": 8
  },
  "20": {
    "good_alignment": 0.6897148703868918,
    "shuffled_alignment": 0.6828598088374956,
    "bad_alignment": 0.6757318114007744,
    "control_alignment": 0.6828598088374956,
    "separation": 0.006855061549396191,
    "mean_norm": 48.803444385528564,
    "good_vector_count": 8
  },
  "21": {
    "good_alignment": 0.6464184129757468,
    "shuffled_alignment": 0.6396733961020421,
    "bad_alignment": 0.6229357771594904,
    "control_alignment": 0.6396733961020421,
    "separation": 0.006745016873704657,
    "mean_norm": 48.14385461807251,
    "good_vector_count": 8
  },
  "22": {
    "good_alignment": 0.5963031478254847,
    "shuffled_alignment": 0.553087385815125,
    "bad_alignment": 0.5480321920103599,
    "control_alignment": 0.553087385815125,
    "separation": 0.043215762010359704,
    "mean_norm": 35.98855793476105,
    "good_vector_count": 8
  },
  "23": {
    "good_alignment": 0.6058523549401865,
    "shuffled_alignment": 0.5663794938621607,
    "bad_alignment": 0.5822232409711159,
    "control_alignment": 0.5822232409711159,
    "separation": 0.023629113969070636,
    "mean_norm": 39.100951075553894,
    "good_vector_count": 8
  },
  "24": {
    "good_alignment": 0.5478258200137136,
    "shuffled_alignment": 0.5149294707050902,
    "bad_alignment": 0.5181461677402814,
    "control_alignment": 0.5181461677402814,
    "separation": 0.029679652273432167,
    "mean_norm": 38.670276165008545,
    "good_vector_count": 8
  },
  "25": {
    "good_alignment": 0.48682379728779746,
    "shuffled_alignment": 0.4557080406632699,
    "bad_alignment": 0.45644295326441214,
    "control_alignment": 0.45644295326441214,
    "separation": 0.03038084402338531,
    "mean_norm": 39.81362462043762,
    "good_vector_count": 8
  },
  "26": {
    "good_alignment": 0.45574681169159914,
    "shuffled_alignment": 0.4219543276299089,
    "bad_alignment": 0.4174681280769133,
    "control_alignment": 0.4219543276299089,
    "separation": 0.03379248406169022,
    "mean_norm": 38.532320737838745,
    "good_vector_count": 8
  },
  "27": {
    "good_alignment": 0.4121232572661005,
    "shuffled_alignment": 0.37333441632716835,
    "bad_alignment": 0.37495475338543494,
    "control_alignment": 0.37495475338543494,
    "separation": 0.03716850388066556,
    "mean_norm": 37.51002275943756,
    "good_vector_count": 8
  },
  "28": {
    "good_alignment": 0.4404921427630179,
    "shuffled_alignment": 0.40890593657507573,
    "bad_alignment": 0.40545166045177394,
    "control_alignment": 0.40890593657507573,
    "separation": 0.03158620618794217,
    "mean_norm": 33.13287591934204,
    "good_vector_count": 8
  },
  "29": {
    "good_alignment": 0.4927855589044278,
    "shuffled_alignment": 0.4794429209058569,
    "bad_alignment": 0.4770722464539417,
    "control_alignment": 0.4794429209058569,
    "separation": 0.013342637998570939,
    "mean_norm": 32.45595705509186,
    "good_vector_count": 8
  },
  "30": {
    "good_alignment": 0.46058521349373577,
    "shuffled_alignment": 0.4340909768826542,
    "bad_alignment": 0.4428232624356434,
    "control_alignment": 0.4428232624356434,
    "separation": 0.017761951058092362,
    "mean_norm": 30.695274829864502,
    "good_vector_count": 8
  },
  "31": {
    "good_alignment": 0.4410987197874356,
    "shuffled_alignment": 0.4225915841186582,
    "bad_alignment": 0.4234012643449104,
    "control_alignment": 0.4234012643449104,
    "separation": 0.017697455442525212,
    "mean_norm": 31.484856128692627,
    "good_vector_count": 8
  },
  "32": {
    "good_alignment": 0.4387464041452377,
    "shuffled_alignment": 0.423585487137611,
    "bad_alignment": 0.42296474081269236,
    "control_alignment": 0.423585487137611,
    "separation": 0.015160917007626673,
    "mean_norm": 32.702613830566406,
    "good_vector_count": 8
  },
  "33": {
    "good_alignment": 0.43003854432879823,
    "shuffled_alignment": 0.42217081386517424,
    "bad_alignment": 0.41507044047574126,
    "control_alignment": 0.42217081386517424,
    "separation": 0.00786773046362399,
    "mean_norm": 33.21807038784027,
    "good_vector_count": 8
  },
  "34": {
    "good_alignment": 0.43858178993175506,
    "shuffled_alignment": 0.43890742875488975,
    "bad_alignment": 0.42682103952137407,
    "control_alignment": 0.43890742875488975,
    "separation": -0.0003256388231346885,
    "mean_norm": 34.93203794956207,
    "good_vector_count": 8
  },
  "35": {
    "good_alignment": 0.440902772837568,
    "shuffled_alignment": 0.4348994302182782,
    "bad_alignment": 0.4325069596239259,
    "control_alignment": 0.4348994302182782,
    "separation": 0.006003342619289809,
    "mean_norm": 38.88060128688812,
    "good_vector_count": 8
  },
  "36": {
    "good_alignment": 0.4325253240443649,
    "shuffled_alignment": 0.4317884073136676,
    "bad_alignment": 0.43035236811238553,
    "control_alignment": 0.4317884073136676,
    "separation": 0.0007369167306973123,
    "mean_norm": 41.04113459587097,
    "good_vector_count": 8
  },
  "37": {
    "good_alignment": 0.4317013421522648,
    "shuffled_alignment": 0.4323235602812597,
    "bad_alignment": 0.43022728564765855,
    "control_alignment": 0.4323235602812597,
    "separation": -0.000622218128994878,
    "mean_norm": 44.271790504455566,
    "good_vector_count": 8
  },
  "38": {
    "good_alignment": 0.4034934915012761,
    "shuffled_alignment": 0.3916223923755,
    "bad_alignment": 0.3888361786400291,
    "control_alignment": 0.3916223923755,
    "separation": 0.011871099125776108,
    "mean_norm": 47.18070459365845,
    "good_vector_count": 8
  },
  "39": {
    "good_alignment": 0.3754652134326289,
    "shuffled_alignment": 0.3561115115618194,
    "bad_alignment": 0.35626252693402227,
    "control_alignment": 0.35626252693402227,
    "separation": 0.019202686498606625,
    "mean_norm": 50.3314311504364,
    "good_vector_count": 8
  },
  "40": {
    "good_alignment": 0.3533902107270012,
    "shuffled_alignment": 0.3206812083544098,
    "bad_alignment": 0.3282436303753555,
    "control_alignment": 0.3282436303753555,
    "separation": 0.025146580351645742,
    "mean_norm": 48.876747846603394,
    "good_vector_count": 8
  },
  "41": {
    "good_alignment": 0.3534223260088095,
    "shuffled_alignment": 0.3198238734540118,
    "bad_alignment": 0.32721594727738446,
    "control_alignment": 0.32721594727738446,
    "separation": 0.026206378731425062,
    "mean_norm": 142.69052982330322,
    "good_vector_count": 8
  }
}

## Invariants
- Order sensitivity: 0.16666666666666669
- Core tokens: ['logic', 'steps', 'given', 'threee', 'false', 'proofs', 'query', 'unless', 'thereo', 'ought', 'checkr', 'termsy', 'sensee', 'issuey', 'whiley', 'axiom', 'nexus', 'though', 'hencey', 'never', 'beyond', 'unwrap', 'syntax', 'reveal', 'modulo', 'indeed', 'pretty', 'secure', 'chain', 'when', 'which', 'must', 'despite', 'initial', 'formal', 'troute', 'theory', 'contra', 'doubt', 'statey', 'belief', 'subset', 'asks', 'specific', 'imply', 'tollens', 'phrasing', 'since', 'might', 'mean', 'still', 'will', 'treat', 'this', 'assumes', 'strong']
- Proxy tokens: []
- Padding tokens: ['matter', 'whereof', 'drives', 'holdss', 'shows', 'maybe', 'among', 'within', 'basis', 'against', 'exactly', 'token', 'each', 'unusual', 'adheres', 'obscure', 'fits', 'above', 'scheme', 'aspect', 'reason', 'scope', 'about', 'locate', 'object', 'always', 'require', 'derive', 'factor', 'system', 'inside', 'remain', 'almost', 'outside', 'cannot', 'premise', 'fallow', 'kernel', 'datum', 'method', 'gauge', 'either', 'versus', 'handle', 'claim', 'appear', 'gains', 'truthy', 'yields', 'events', 'bounds', 'fading', 'limits', 'user']

## Causal
- Summary: {
  "context_collision_anchor": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 1.1960720928618684,
    "mean_logit_cosine_shift": 0.03838746783624611,
    "mean_attention_to_control": null
  },
  "best_anchor_eval": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 1.3099811925785616,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_class_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 20.72386757854838,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_bad_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 9.052549785003066,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "recover_with_shuffled_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 9.054938763496466,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "remove_class_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 1.254084107116796,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  },
  "project_out_class_delta": {
    "count": 2,
    "pass_rate": 0.0,
    "mean_score": 0.0,
    "mean_latency_s": 1.2698016839567572,
    "mean_logit_cosine_shift": null,
    "mean_attention_to_control": null
  }
}

## Compression / Outlier
- Summary: {
  "good": {
    "layer_norms": {
      "22": 35.98855793476105,
      "27": 37.51002275943756
    },
    "norm_concentration": 0.5103503007153756,
    "stability": 0.6970092803239822,
    "top_abs_mass_ratio": 0.0669021513313055
  },
  "bad": {
    "layer_norms": {
      "22": 33.00078797340393,
      "27": 33.92010307312012
    },
    "norm_concentration": 0.5068686704954143,
    "stability": 0.7192659005522728,
    "top_abs_mass_ratio": 0.0703947339206934
  },
  "shuffled": {
    "layer_norms": {
      "22": 37.197479248046875,
      "27": 36.516995429992676
    },
    "norm_concentration": 0.5046156729802819,
    "stability": 0.7213249653577805,
    "top_abs_mass_ratio": 0.07091540656983852
  },
  "good_minus_controls": {
    "norm_concentration_delta": 0.0034816302199612847,
    "stability_delta": -0.024315685033798218,
    "top_abs_mass_ratio_delta": -0.00401325523853302
  }
}

## Bad Anchors
- `90a0c5919ed5`: pass_rate=0.5 alignment=0.9359368319775329
