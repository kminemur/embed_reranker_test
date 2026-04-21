# Minimum Wages 2022 Test (Embedding + Reranker)

This small test script checks output scores from:

- OpenVINO/bge-base-en-v1.5-int8-ov
- OpenVINO/bge-reranker-base-int8-ov

using the question:

"What is the minimum wages order for 2022 in Malaysia?"

## Quick answer to the question

Based on P.U. (A) 1406-1409 (Minimum Wages Order 2022):

- The order is cited as the Minimum Wages Order 2022, effective 1 May 2022.
- Main minimum wage: RM1,500 monthly (with corresponding daily/hourly rates).
- For certain employers with fewer than 5 employees (transition period, 1 May 2022-31 Dec 2022):
	- RM1,200 in City/Municipal Council areas
	- RM1,100 outside those areas
- Domestic servants are excluded.
- The Minimum Wages Order 2020 is revoked.

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run:

```bash
python test_bge_outputs.py
```

The script prints, for each candidate passage:

- embedding cosine similarity (`embed_cos`)
- reranker probability-like score (`reranker_sigmoid`)

and sorts results by reranker first, then embedding score.

## Expected
```
=== Iteration 1/5 ===
Question: What is the minimum wages order for 2022 in Malaysia?

[1] embed_cos=0.8182  reranker_sigmoid=0.7119
P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and paragraph 6 effective for 1 May 2022 to 31 Dec 2022.

[2] embed_cos=0.7855  reranker_sigmoid=0.4174
P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, with daily and hourly rates listed. It applies to employers with five or more employees and professional activity employers under MASCO.

[3] embed_cos=0.6975  reranker_sigmoid=0.4047
P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees and area-based rates for 1 May 2022 to 31 Dec 2022: RM1,200 in city/municipal council areas and RM1,100 outside.

=== Iteration 2/5 ===
Question: What is the minimum wages order for 2022 in Malaysia?

[1] embed_cos=0.8182  reranker_sigmoid=0.7119
P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and paragraph 6 effective for 1 May 2022 to 31 Dec 2022.

[2] embed_cos=0.7855  reranker_sigmoid=0.4174
P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, with daily and hourly rates listed. It applies to employers with five or more employees and professional activity employers under MASCO.

[3] embed_cos=0.6975  reranker_sigmoid=0.4047
P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees and area-based rates for 1 May 2022 to 31 Dec 2022: RM1,200 in city/municipal council areas and RM1,100 outside.

=== Iteration 3/5 ===
Question: What is the minimum wages order for 2022 in Malaysia?

[1] embed_cos=0.8182  reranker_sigmoid=0.7119
P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and paragraph 6 effective for 1 May 2022 to 31 Dec 2022.

[2] embed_cos=0.7855  reranker_sigmoid=0.4174
P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, with daily and hourly rates listed. It applies to employers with five or more employees and professional activity employers under MASCO.

[3] embed_cos=0.6975  reranker_sigmoid=0.4047
P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees and area-based rates for 1 May 2022 to 31 Dec 2022: RM1,200 in city/municipal council areas and RM1,100 outside.

=== Iteration 4/5 ===
Question: What is the minimum wages order for 2022 in Malaysia?

[1] embed_cos=0.8182  reranker_sigmoid=0.7119
P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and paragraph 6 effective for 1 May 2022 to 31 Dec 2022.

[2] embed_cos=0.7855  reranker_sigmoid=0.4174
P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, with daily and hourly rates listed. It applies to employers with five or more employees and professional activity employers under MASCO.

[3] embed_cos=0.6975  reranker_sigmoid=0.4047
P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees and area-based rates for 1 May 2022 to 31 Dec 2022: RM1,200 in city/municipal council areas and RM1,100 outside.

=== Iteration 5/5 ===
Question: What is the minimum wages order for 2022 in Malaysia?

[1] embed_cos=0.8182  reranker_sigmoid=0.7119
P.U. (A) 1406 defines the Minimum Wages Order 2022 under Act 732. It starts on 1 May 2022, with paragraph 5 from 1 Jan 2023 and paragraph 6 effective for 1 May 2022 to 31 Dec 2022.

[2] embed_cos=0.7855  reranker_sigmoid=0.4174
P.U. (A) 1407 sets minimum wages from 1 May 2022 at RM1,500 monthly, with daily and hourly rates listed. It applies to employers with five or more employees and professional activity employers under MASCO.

[3] embed_cos=0.6975  reranker_sigmoid=0.4047
P.U. (A) 1408 and 1409 describe rates for employers with fewer than five employees and area-based rates for 1 May 2022 to 31 Dec 2022: RM1,200 in city/municipal council areas and RM1,100 outside.
```
