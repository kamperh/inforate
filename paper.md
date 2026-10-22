---
title: "On the information rate of phones, syllables and words"
author: "[Herman Kamper](https://www.kamperh.com/)"
---

## Abstract

Which linguistic units minimise the bitrate required to encode speech content: phones, syllables, words or maybe larger units? This question has implications for theories of human language processing and the development of speech and language systems. Using a transcribed corpus of 960 hours of English audiobooks, I compare the information rates of phones, syllables, words, phrase-like units and intermediate tokenisations. While all units are equivalent under a perfect language model, it makes sense to compare them using these limited-capacity models. The lowest bitrates are achieved using words, with slightly longer units sometimes performing best. The results show that, when modelling capacity is limited, units larger than syllables but smaller than long multi-word phrases provide the most efficient representation in terms of information rate.



## Introduction

Suppose I have a spoken utterance, and I want to send you the content of that utterance, i.e.\ the speech sounds in it. I write the utterance down phonetically, then I encode the phones in bits, you receive the bits, and (because we agreed on an encoding scheme), you can decode the bits and know what phones were in the utterance. For example, if the person said ... . This will take a certain number of bits. We will set our encoding scheme based on the conditional probabilities of the phones, so that we use the channel optimally.

But, rather than phones, we can also decide to tokenise the utterance into other different types of units. The goal is still for me to send you the phonetic content, but now we decide that, instead of phones, I first convert the sequence into syllables (e.g. ...). Now we decide on a syllable-to-bit encoding scheme, again I send you the bits, and you can decode the bits into syllables, and then decode the syllables into phones. Again we use conditional probabilities to assign bit sequences to (now) syllable types. A different tokenisation, but I can communicate the phones exactly. Is this more efficient than phones?

Rather than syllables, we can also decide on a different encoding scheme: let's create a tokenisation into words (where here each "word" corresponds to a unique phonetic realisation). Or maybe we can decide on even longer phrase-like units, such as ... . By now you probably get the picture.

This paper asks which levels (phones, syllables, words, units longer than words) are the most efficient for communicating linguistic content. All experiments are done on transcribed phonetic sequences (obtained from forced alignments) on English data. Why do this comparison?

For me, one reason is that I want to know the relative efficiencies of these different linguistic tokenisations---this is scientifically useful to know. And these aren't reported in the literature (or at least not consistently). You find people reporting bitrates like 50 bps for phones (Lopes and Aguiar 2010; Flanagan et al. 2014) and 39 bps for syllables (Coupé et al. 2019).^[In defence of the latter, their goal was not to produce *the* bitrate of syllables.] But a closer reading reveales that these are not really comparable (because the definitions of bitrates are not consistent, and the data on which they are calculated are not the same). And information rates would always depend on the content.

A second reason is that I come from the zero-resource speech processing community. To be clear: none of this paper deals with speech; all experiments are on transcribed phonetic sequences. But the question is still worth asking: in this ideal setting, are some units more efficient than others? This is particularly relevant to the zerospeech community, because some of us work explicitly on finding phone-like units (), while others have focussed on finding words (), while recently there has been a big push to syllable-like units (). Some early and more recent studies show that finding recurring phrases is even possible (). But, under a fixed modelling budget, are some units more efficient than others?

* * *

**Everything is at the phone level.** 

**Maybe but in conclusion rather?** Understanding which linguistic units support efficient encoding is relevant to theories of human language processing. If language processing is shaped by pressures for efficient information transmission, the choice of representational units may play a role. Comparing phones, syllables, words and larger units provides a way to investigate how the granularity of linguistic representations affects information rate. Crucially, however, the relative efficiency of these units may depend on the learner's or processor's modelling capacity: representations that are equivalent in principle may differ in practice when learning and processing resources are limited. Quantifying these differences offers a computational perspective on the relationship between linguistic representations, information efficiency and finite modelling capacity. I'd be cautious about claiming that lower bitrate directly predicts human processing preferences. Your experiments establish a result about language models, not human cognition. The connection is strongest if you frame the work as providing a computational perspective on a question relevant to theories of human language processing.

* * * 

Why are we looking for words? Maybe some other unit is "optimal" in some sense? Maybe syllables or smaller units like phones. Or maybe much bigger units, like phrases?

So I thought: what are the bitrates for these different units? Doing a quick literature search, you find people reporting bitrates like 50 bps for phones (Lopes and Aguiar 2010; Flanagan et al. 2014) and 39 bps for syllables (Coupé et al. 2019). But a closer reading reveales that these are not really comparable (because the definitions of bitrates are not consistent, and the data on which they are calculated are not the same).

This is a quite boring talk, but I just want to show you the journey. If I were a bit more humble, I would have called the paper "The information rate of phones, syllables and words in English".



## Information rate (bitrate) (short)

The definition of information rate (or bitrate) in bits per second is (Coupé et al. 2019):
$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{bits}}{\text{duration}} \\
    &= \frac{\text{\#units}}{\text{duration}} \times \text{bits per unit} \\
\end{aligned}
$$
To know the average bits per unit, we need to decide on an encoding scheme. If we use entropy coding, where the number of bits are chosen based on the probability of the unit, then the bits per unit would be given by the entropy of the unit distribution (Shannon 1948; Clover and Thomas 2006; Sanderson 2026). But, to do this, we would need to know the true unit probabilities $P(w_t \mid w_{1:t-1})$. We don't have this; instead we have a model $P_{\theta}(w_t \mid w_{1:t-1})$. Under this model, we can then estimate the number of bits on held out data (of $T$ points), using:
$$
\text{bits per unit} = - \frac{1}{T} \sum_{t = 1}^T \log_2 P_{\theta}(w_t \mid w_{1:t-1})
$$
This is technically the estimated cross-entropy rate when we use model $P_{\theta}$ to encode units coming from the true distribution $P$. From this, we can get the bitrate:
$$
\text{bitrate}
= \frac{\text{bits}}{\text{duration}}
= -\frac{1}{D} \sum_{t = 1}^T \log_2 P_{\theta}(w_t \mid w_{1:t-1})
= -\frac{1}{D} \log_2 P_{\theta}(w_{1:T})
$$
where $D$ is the duration in seconds.^[If you wanted to keep this simple and knew everyone had a background in information theory, you could also just say this is the sum of the Shannon surprisal (Shannon 1948) divided by the duration.]

**Idea:** Measure the bitrate of different units (phones, syllables, words) as a way to quantify encoding efficiency. Concretely, we ask what $w$ should be to give us low bitrates.



## But units don't matter: they all have the same (ideal) bitrate

$$
\begin{aligned}
    P(\texttt{New York is beautiful})
    &= P(\texttt{New}) \cdot P( \texttt{York} \mid \texttt{New}) \cdot P( \texttt{is} \mid \texttt{New York}) \cdots \\
    &= P(\texttt{New York}) \cdot P( \texttt{is} \mid \texttt{New York}) \cdots
\end{aligned}
$$

But how does this make sense (given our intuition about merging `New York` since they always co-occur -- in this fictitious world)?

Let's say `York` truly always follow `New` (and no other word ever follows `New`). Then:
$$
    P( \texttt{York} \mid \texttt{New}) = 1
$$
And so $P(\texttt{New York}) = P(\texttt{New})$. Stated in information theoretic terms: it takes zero extra bits to transmit `York`. So whether you glue it onto `New` or leave it on its own, it costs the same number of bits.^[It is now for the first time how I see the amazingness of the connection between information theory and probability theory: you can think about probabilities as just communicating information, which you can think of in terms of bits.] More formally, the different ways of tokenising are simply different ways of defining random variables on the same underlying sample space. If I were able to obtain the true probabilities, then I could even split it up into characters, and the result should still be the same: $P(\texttt{N}) \cdot P(\texttt{e} \mid \texttt{N}) \cdot P(\texttt{w} \mid \texttt{Ne}) \cdots$.

So apparently this question of the optimal number of bits with different encoding units is silly. They are all the same. Or are they?

The assumption above is that we know the true probability distribution $P$, the actual distribution governing the real world. But no-one (except the Creator) actually knows these probabilities. So we always have a model $P_{\theta}$. And this model would be tied to a particular type of unit. And then the above doesn't happen: it now matters how you tokenise your input. If you had a really really powerful model that were really able to almost perfectly estimate the probabilities, then it doesn't matter -- they will all end up having the same bitrate. But under limited capacity, it does matter which units you use.

So we refine our research question: What are the most efficient unit for encoding speech content *under limit capacity*?



## Models

To limit the capacity, we choose (and then limit) some model type. In this paper, we consider two types of models: n-gram language models (Chen and Goodman 1999; Jurafsky and Martin 2026) and transformer-based neural language models (Vaswani et al. 2017).

### N-gram models

I use absolute discounting (Chen and Goodman 1999, \S2.6) in which higher-order n-gram language models are interpolated with lower-order models. Let's look at the highest order $n$, where we get the probability $P_n(w_t \mid w_{t - n + 1 : t - 1})$. To simplify the notation, we use $h = w_{t - n + 1 : t - 1}$ to denote the preceding symbols. We then have:
$$
    P_n(w_t \mid w_{t - n + 1 : t - 1})
    = P_n(w_t \mid h)
    = \frac{\max \{ c(h, w_t) - d, 0\}}{c(h)} + (1 - \lambda_{h})P_{n - 1}(w_t \mid h')
$$
where $h' = w_{t - n + 2 : t - 1}$ drops the first symbol from $h$. For these probabilities to sum to one, we need
$$
    1 - \lambda_{h} = \frac{d}{c(h)} N_{1+}(h\cdot)
$$
where $N_{1+}(h\cdot) = |\{w : c(h, w) > 0\}|$ is the number of distinct continuations
observed after $h$.
We therefore get
$$
    P_n(w_t \mid h) = \frac{ \max \{ c(h, w_t) - d, 0\} } {c(h)} +  \frac{d N_{1+}(h\cdot)} { c(h) } P_{n - 1}(w_t \mid h')
$$
This recurses down to the unigram case:
$$
    P_1(w_t) = \frac{ \max \{ c(w_t) - d, 0\} } {T} + \frac{Vd}{T} P_0 \quad, \quad P_0 = \frac{1}{V + 1}
$$
The $+1$ in the denominator of $P_0$ reserves probability mass for out-of-vocabulary `<unk>` symbols (I will explain this below).

Note that above I wrote a single discount value $d$ for all orders, but we actually set $d$ separately for each order. Following the recommendation of Ney et al.\ (1994), we use
$$
    d = \frac{n_1}{n_1 + 2n_2}
$$
where $n_1$ and $n_2$ are the number of distinct n-grams ocurring exactly once and exactly twice in the training data (at this order). If these counts are zero, we fall back to a value of $d = 0.75$ (a value I did not tune). I also clamp the discount to the range $d \in [0.1, 0.9]$ to avoid very strange values.

**Out-of-vocabulary tokens in the n-gram models.** When dealing with words and syllables, we have to deal with tokens that are not seen durint training. These are all mapped to `<unk>`. But then we back off to a spelling model, paying bits to spell out the token. The spelling will take different forms depending on what is spelled out, but for now you can just think of a word being spelled out as letters.

What technically happens is that when we hit `<unk>` in the $P_0$ above, it doesn't use $\frac{1}{V + 1}$ (as it does for all the seen tokens). Instead it backs off to a spelling model $P_\mathrm{spell}(\mathrm{spelling}(w_t))$. This model is itself an absolute discounted n-gram using the interpolation as above, but trained on spelled out tokens. (I always use a third-order spelling model.) So, if we wanted to be a bit more formal, I would write
$$
    P_0(w_t) =
    \begin{cases}
    \frac{1}{V + 1} & \text{if\ } w_t \text{\ is seen} \\
    \frac{1}{V + 1} P_\mathrm{spell}(\mathrm{spelling}(w_t)) & \text{if\ } w_t \text{\ is unseen} \\
    \end{cases}
$$
How I visualise this is that $\frac{1}{1 + V}$ mass is allocated to each of the $V$ seen words. For unseen words, the remaining $\frac{1}{1 + V}$ mass is spread out, but non-uniformly based on the probability of their constituent spellings (i.e.\ a word that is easier to spell get a larger chunk of the $\frac{1}{1 + V}$-`<unk>`-mass). The spelling model ends with an end-of-word symbol (otherwise the probabilities would not be normalised).

**Implementation.** In development experiments, I varied the amount of training data (25%, 50%, 75%, 100%) and the n-gram order (from 1 to 5) for all six the different units (below). For most of the units, trigrams worked the best, and I therefore use this throughout. This is then our fixed n-gram budget.^[The only exception was phone models, where a 5-gram worked better. I give results for all the models with 5-grams as well in the Appendix.]

### Neural models

I use a decoder-only autoregressive transformer language model as the base. The architecture is a modification of the OPT-125M architecture (Zhang et al. 2022). I specifically reduced the model size so that the body (excluding the output layer predicting units) has approximately 20M parameters -- this is kept constant regardless of the unit on which the model is trained. In the tables I therefore call the model OPT-20M. Each model is a standard causal transformer using stacked blocks with masked multi-head self-attention and a feed-forward layer, pre-layer-norm, ReLU activations, learned positional embeddings, and tied input and output embeddings. The variant I use has 6 layers with 512 hidden dimensions, 8 heads, feedforward networks with 2048 dimensions and a context window of 2048 tokens. This gives all the models the same body of roughly 20M parameters.^[The number of tokens per parameter differ based on the unit, but for phone units this gives 1.69 tokens per parameters. I could probably have gotten away with even smaller models, but I didn't tune this: I picked 20M and moved on with life. But later you see that I do also check bitrates with a much bigger separately trained Qwen model as a sanity check. These results give me some confidence that the budget choice here is reasonable.] What does differ between the models depending on the target unit is the output layer. More detail is given on the different units below, but the output of a phone neural language model only has 39 classes (together with sentence and pad markers) while the word model has roughly 58k classes.^[You could argue that I had to balance model sizes taking the (sometimes substantial) output layer into account. The problem is that it is not at all clear how this should be done. Also note that the word and all the word-BPE models have roughly the same number of output classes because of the way the vocabularies are constructed (details to follow).]

**Out-of-vocabulary tokens in the neural models.** As with the n-grams, sometimes we have to deal with tokens that have not been seen in the training data. We follow is similar spelling-out strategy here. Let us use word units as an example. To deal with this, unseen words are broken up into their constituent letters. These letters get their own markers, e.g. `aardwolf` gets broken up as `~a ~a ~r ~d ~w ...` (this means `a` as a word is distinct from `~a` which is a word-part). All training words not in the vocabulary are broken up this way. But I also randomly (2% of the time) break a word up into its constituent parts (whether it is seen or not). This is so that the model sees more broken up words at training time and is then better able to deal with unseen words at evaluation time.

**Implementation.** All the neural models are trained with an effective batch size of 40 on a single NVidia RTX3070 (8Gb). I use AdamW optimisation (Loshchilov and Hutter 2017) with a linear warmup for 8% of the steps to a maximum learning rate of $\text{1}\cdot\text{10}^{-3}$ followed by a cosine decay to $\text{2}\cdot\text{10}^{-5}$ over the remaining steps. Models are trained for approximately 20 epochs; the step sizes are changed based on the unit granularity to ensure that the equivalent of roughly 20 passes is made over the encoded dataset. These values were chosen based on the original recipe from (Visser et al. 2026). The only change was to raise the maximum learning rate from $\text{5}\cdot\text{10}^{-4}$ to $\text{1}\cdot\text{10}^{-3}$; this slightly improved final performance across all the units (but the trends were always the same). I also experimented with training for more epochs, but this didn't help any of the units.



## Data and units

**How many units (in terms of tokens)?**

I train on LibriSpeech 960h and evaluate on dev-clean (4.5 hours of speech). All the experiments are therefore conducted in English. For future work, it would be essential to also repeat this analysis on other languages.

**Phones.** Phone-level forced aligments for the data were produced using Montreal Forced Aligner (McAuliffe et al. 2017) using an ARPABet phone set. Stress markers are removed, leaving a set of 39 phones.

**Syllables.** As in (Visser et al. 2026), a syllable tokenisation is obtained from the phone sequences using a rule-based method (Gorman 2013). Word boundaries are not taken into account, i.e.\ syllables can span over words. When encountaring an out-of-vocabulary syllable, the model backs off to a "spelled out" version of the syllable, where "spelling" here means using the constituent phones. (This spelling model is used in both the n-gram and neural models.)

**Words.** For the word case, out-of-vocabulary words are spelled out as characters. For the neural model, only words that occur at least twice in the training data are used in the output vocabulary, resulting in 58k classes. Words occurring only once or not at all in the training data is therefore spelled out.

**Word-BPE.** Maybe the most efficient units are something larger than words, like phrases or common-multi-word combinations. To test this, I use BPE, but not the standard BPE where subwords are built up by merging commonly co-occurring characters (Gage 1994; Sennrich et al. 2016). (Below I do do a sanity check using standard subword-BPE.) Rather, here, multi-word combinations are found by merging at the word-token level. New types like `of the` are therefore added to the vocabulary. I again constrain the starting vocabulary to words occurring at least twice (58k types), and then add the new merged word types. I consider 200, 1000 and 5000 merges. Unseen words are spelled out as characters, as with the plain word model.

The most common merges word the word-BPE units are the following:

- 'of' + 'the'  ->  'of the'
- 'in' + 'the'  ->  'in the'
- 'to' + 'the'  ->  'to the'
- 'and' + 'the'  ->  'and the'
- 'on' + 'the'  ->  'on the'
- 'it' + 'was'  ->  'it was'
- 'to' + 'be'  ->  'to be'

For the 5000-merge case, the longest 15 merges in terms of number of tokens are:

- 'on the other side of the'
- 'on the part of the'
- 'in the midst of the'
- 'in the middle of the'
- 'at the foot of the'
- 'at the end of the'
- 'and at the same time'
- 'with an air of'
- 'what is the matter'
- 'what do you think'
- 'what do you mean'
- 'up and down the'
- 'to go to the'
- 'the rest of the'
- 'that there was no'

- Add vocabulary sizes to paper (with and without min-count)



## Results

The goal is to see what unit granularity has the best information rate under a constrained modelling budget. We look at phones, syllables, words, and different word-BPE tokenisations (at 200, 1000 and 5000 merges). This gives a total of six units for comparison.

**Notes on development experiments on single units.** Bitrates for syllable units as the amount of data is varied:

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 46.22 | 35.24 | 32.98 | 33.06 | 33.25 | - |
| 50% | 46.21 | 34.91 | 32.28 | 32.30 | 32.51 | - |
| 75% | 46.20 | 34.78 | 31.94 | 31.91 | 32.13 | - |
| 100% | 46.20 | 34.68 | 31.68 | 31.61 | 31.82 | 28.41 |

This trend was the same for all models, except phones, for which a 5-gram was the best (38.45 for 4-gram vs 34.93 bps for 5-gram when using 100% of the data). But I wanted a fixed order for all the models, so I picked trigrams. (And even if I did use a 5-gram just for phones and a 3-gram for all the others, the conclusions would still be the same. The problem with this is that would allow for a changing budget, which violates our research question. In the Appendix I give the order 5 results for all the models.)

**Main results: comparison between units.**

![Bitrates for the different units per model type. The dark bar gives the best performance per model type. The error bars are runs across multiple seeds.](docs/fig/unit_bars.pdf){width=85%}



## Sanity checks

- Samples from the neural word model given the prompt `the cat sat ...`

    - `the cat sat for some time gazing on the ground in the dark woods and she knew that she had seen it was true that she had had never seen it before`
    - `the cat sat up in the cradle and the cat who had been watching for her life in the eyes of her cat but the cat soon came to look about her`
    - `the cat sat down before him i have not been in love with her but you will be glad to see her and that you will not be so dear to me`
    - `the cat sat there in her bedroom`
    - `the cat sat in her bed`
    - `the cat sat up in a hole in the trunk of a tree on the floor and on ~t ~h ~e sill of the bed there was a soft rustling sound and the dog barked out of his ears a very hard and the dog barked at the dog's throat then he stopped short for the dog to catch hold of his dog in a dog cart`

- Multiple seeds: in figure

- Bootstrap confidence intervals

    - Use 10,000 bootstrap samples, sampling with replacement at the speaker level (sample 40 speakers with replacement) (Bisani and Ney 2004)
    - All of the pair-wise comparisons are statistically significant at the 99.75\% level, except for one.
    - The one exception is the $n$-gram word model when compared to the BPE-1000 model, where the difference is not significant (even at the 95\% level).
    - But the improvement of n-gram BPE-1000 over BPE-5000 is significant.

- Subword BPE: sanity check. Is the bitrate similar to syllable?

    - Merge characters to get 2000 types
    - Bitrate is 28.06 $\pm$ 0.11 with an OPT-20M model
    - So not as good as word (27.49) and better than syllable (28.38)

- Qwen model

    - We don't know how close we are to the actual entropy bitrate (because we don't have a perfect model). But we can look at the bitrate using substantially bigger models than those used here. For this I use two Qwen2.5 models (Qwen Team, 2024).

    - Setup:

        - These models are trained on roughly 18T tokens of text. They use a byte-pair BPE tokeniser with an output vocabulary of about 150k types.
        - Qwen conditions on a lowercase, unpunctuated primer drawn from train-clean-100, prepended to every utterance as shared context and never itself scored. This is just to get the model into the head-space of this domain. Priming helps for both models below.

    - I use two models to measure bitrate on my data.

        - Qwen2.5-1.5B: 25.90
        - Qwen2.5-3B: 25.26

    - The best bitrate above with the models trained on LibriSpeech is the word OPT-20M model giving a bitrate of 27.46.

    - I think this shows that my models are in the right ballpark. As model size and amount of data scales, we should expect the different units to give similar bitrates. But the point of this study was to do a comparison under the same budget. (This was just for sanity checking.)



## Discussion

We are not encoding the actual "message", which probably has much fewer bits, but also not the actual speech (speaker, emotion, channel), which probably would cost way more bits.

Just because encoding is more efficient, it doesn't necessarily mean generation is going to be better (it almost certainly does not). It is fine to spend a large number of bits on long sequences that occur once or twice, but if you accidentally generated these during inference in the wrong context, the penalty could be very unintelligible output.



## Conclusion

I do not claim that words are the most efficient units (especially given that the concept of what a word is is contested, and also quite language dependent). What I would claim is that the most efficient unit is something bigger than phones or syllables.



## Limitations

In some of the derivations I assmed a one-to-one mapping, for instance, from words back to phones (and vice versa), but this is not exactly true: without knowing the boundaries, it is difficult to get from phones to words. Another problem is in backing off to the spelling model in different ways; with the word models we back off to spelling via characters, while with the syllable models, the spelling model is via the constituent phones.

For the LLMs, I matched parameter size on the bodies of the models, not the output vocabularies. This means that the word-based models in effect has many more parameters than the other models. But I feel okay about this, since I matched the vocabularies of the word-BPE models, so they are at least all the same size. I can also not see how you would make the parameters actually line up e.g. between phone and word models.

Related to the point under discussion about the “message”, this work really looked at encoding efficiency purely from an information theory standpoint (under a fixed budget). The goal was to show that – from this viewpoint – some units might be better for modelling than others. But maybe it would be better to explicitly consider what humans do through a psycholinguistic lens. As one example, Oh and Schiler (2024) considered how different subword-BPE tokenisations on text influence the correlation of estimated surprisal with human processing times.

The claim here seems to be that something at the word or super-word level is more efficient (under a constrained budget). But the concept of a “word” is itself controversial, and will differ across languages (all of the above was done on English). This is true, and the work should be extended to other languages. Nevertheless, I think the results show that something larger than a syllable is more efficient than units like syllables or phones, and I suspect that will hold even for other languages.

I am worried that the word level model does better, given that everyone else trains BPE level models.

I am really worried I missed something very dumb that will completely invalidate all of this. One thing in particular that worries me is that the word-based model outperforms a subword-BPE model, and no one uses a full word-based model for textual language models. I suspect this has to do with scale (they train on way more data). But I think my findings is good given the framing that this comparison is done *under a constrained modelling budget*.



## How I used AI

Three iterations of this project:

1. Middle 2025: First attempt on multiple languages (failed)

2. Start 2026: "Claude, look at what I did and fix it" (failed)

3. End of 2026: Current version (okayish)


Iteration three ended up with:

- `reproduce.md`
- `design.md`
- `results.md`
- `CLAUDE.md`
- Planning documents.

Writing becomes important again: as I wrote, I realised I didn't know what Claude was doing. I then actually found mistakes in the code (e.g. the spelling out models in the neural case; end-of-utterance tokens in the neural case; end-of-word symbol in the $n$-gram case).

AI policy:

- Write the first paper draft without any AI.

    - Writing is thinking
    - The crappy first draft is really important (the most important?)
    - Afterwards, AI can help
    - Anchoring affect

- Two analogies:

    - Barack Obama (Sal Kahn)
    - A book editor

- AIs don't poo

    - Psychologists

- How would the reader feel if they knew what you did? (Rachel Gilson, Silicon Spiritualities)

- Each of us our own policy:
    
    - My personal AI policy


## Acknowledgements

- Nicol Visser - 3Blue1Brown, code
- Benjamin van Niekerk
- Simon Malan for the alignments



## References

"Information-theoretic analysis of tokenisation and text predictability in South African languages," in *SACAIR*, 2026.

Byung-Doh Oh, William Schuler, “The Impact of Token Granularity on the Predictive Power of Language Model Surprisal”, https://arxiv.org/abs/2412.11940. 



# Supporting material and notes

## Bitrates (long)

Let us start with the definition of bitrate that we will end with. We fit some model with parameters $\theta$ on training data, and then we calculate probabilities on unseen evaluation data consisting of $T$ tokens. We then have the following definition for bitrate (what Coupé et al. (2019) calls information rate) in bits/sec:
$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{bits}}{\text{duration}} \\
    &= \frac{- \sum_{t = 1}^T \log_2 P_{\theta}(w_t | w_1, w_2, \ldots, w_{t - 1})}{D}
\end{aligned}
$$

That is intuitive (if you watch 3Blue1Brown), but let us build up to it a little bit more formally.

$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{bits}}{\text{duration}} = \frac{\text{\#units} \times \text{bits per unit}}{\text{duration}} \\
    &= \frac{T \times \text{entropy}}{D}
\end{aligned}
$$
This works because of what entropy is:
$$
    \text{entropy} = \text{bits per unit} = -\sum_{w} P(w) \log_2 P(w)
$$
Unpacked a little bit more: entropy gives you the lowest average bits per unit that you can get for some data from a distribution $P(w)$. If you assign bits to symbols in this way, you can call it *entropy coding* (Cover and Thomas 2006).

Great, so we can calculate bitrate! The problem is we don't know what $P(w)$ -- the real distribution -- actually is. So we make a mistake: we use a *model* $P_{\theta}(w)$ to do our coding. This is not as good as using the actual distribution (but that's what we have, and we can spend a lot of time trying to make the model as good as possible). But now you won't actually use entropy in your calculation of bitrate, but rather cross-entropy:
$$
    \text{cross-entropy} = -\sum_{w} P(w) \log_2 P_{\theta}(w)
$$
This tells you on average how many bits we need per unit when encoding data from $P(w)$ with our (wrong) model $P_{\theta}(w)$.

Great, so we can calculate bitrate! The problem is we still don't know what $P(w)$ is, so we still can't calculate the cross-entropy. But we can estimate it (on some unseen evaluation data preferably):
$$
    \text{estimated cross-entropy} = \frac{1}{T} \sum_{t = 1}^T \log_2 P_{\theta}(w_t)
$$
Why is this an estimate of the cross-entropy? If you have seen the Monte Carlo sampling method before, this just feels natural. If you haven't seen this before, it could freak you out, but stay calm: spend a while working through ([Resnik and Hardisty 2010](https://www.ccs.neu.edu/home/vip/teach/DMcourse/5_topicmodel_summ/sampling/LAMP-TR-153.pdf), Sec. 1.2.2). Or, for now, just believe me.

Great, so we can calculate bitrate! No, because up to now we made the fatal flaw of thinking that every $w_t$ is independent of all the others, which it really isn't. We actually made this mistake right at the start: we shouldn't use entropy, instead we should use *entropy rate* because we are looking at sequences of events (formally, we are looking at random processes, look at (Clover and Thomas 2006) again). Luckily, we can modify our equation to give us the estimated cross-entropy rate:
$$
    \text{estimated cross-entropy rate} = - \frac{1}{T} \sum_{t = 1}^T \log_2 P_{\theta}(w_t | w_1, w_2, \ldots, w_{t - 1})
$$

Great, so we can calculate bitrate! Yes, now we actually can:
$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{bits}}{\text{duration}} \\
    &= \frac{T \times \text{estimated cross-entropy rate}}{D} \\
    &= - \frac{T}{D} \times \frac{1}{T} \sum_{t = 1}^T \log_2 P_{\theta}(w_t | w_1, w_2, \ldots, w_{t - 1}) \\
    &= -\frac{1}{D} \sum_{t = 1}^T \log_2 P_{\theta}(w_t | w_1, w_2, \ldots, w_{t - 1})
\end{aligned}
$$
What do you know: that's where we started.

I want to quickly mention two other ways some clever people calculate bitrate. In both cases I think this is sensible proxies for the bitrate above, but it is slightly different.

In most of my group's work, we use the way set in the ZeroSpeech Challenge (van Niekerk, 2024):
$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{\#units} \times \text{bits per unit}}{\text{duration}} \\
    &= - \frac{ T }{D} \sum_{w} P_{\theta}(w) \log_2 P_{\theta}(w)
\end{aligned}
$$
Here $P_{\theta}(w)$ is the unigram (first-order) probabilities from the model that you fitted. This actually makes sense; you are looking at the entropy of your estimated distribution. My definition of bitrate above matches this exactly if you (a) assume that each token is independent of the others, and (b) the data that you are evaluating your bitrate on is the same data on which you got your unigram estimates from.

Coupé et al. (2019) does something very similar, but they use a second-order model that does take the single preceding token into account:
$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{\#units} \times \text{bits per unit}}{\text{duration}} \\
    &= -\frac{T}{D} \sum_{w_{t - 1}} \sum_{w_{t}} P_{\theta}(w_{t - 1}, w_t)
        \log_2 \frac{ P_{\theta}(w_{t - 1}, w_t) }{ P_{\theta}(w_{t - 1})}
\end{aligned}
$$
Again this matches my definition if we make a second-order modelling assumption and calculate bitrate on the training data.

I think if you are comparing things, the important thing is just to calculate bitrate consistently. You cannot compare bitrates calculated the ZeroSpeech way to bitrates calculated the Coupé way.


## Titles

- The most efficient units for encoding speech content
- On the spoken information rate of different units
- On the information rate of different speech units
- On the information rate of different units
- On the information rate of different spoken units