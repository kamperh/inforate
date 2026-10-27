---
title: "On the information rate of phones, syllables and words"
author: "[Herman Kamper](https://www.kamperh.com/)"
---


## To-do

- [ ] Consistent use of tokens/types/units/tokenisation vs. classes.
- [ ] Consistent notations for words (maybe italics?) and ARPABet phones. Maybe without quotes?



## Abstract

Which linguistic units minimise the bitrate required to encode speech content: phones, syllables, words or maybe larger units? This question has implications for theories of human language processing and the development of speech and language systems. Using a transcribed corpus of 960 hours of English audiobooks, I compare the information rates of phones, syllables, words, phrase-like units and intermediate tokenisations. While all units are equivalent under a perfect language model, it makes sense to compare them using these limited-capacity models. The lowest bitrates are achieved using words, with slightly longer units sometimes performing best. The results show that, when modelling capacity is limited, units larger than syllables but smaller than long multi-word phrases provide the most efficient representation in terms of information rate.



## Introduction

Suppose I have a spoken utterance, and I want to send you the content of that utterance, i.e.\ the speech sounds that it contain. I write the utterance down phonetically, then I encode the phones in bits, you receive the bits, and (because we agreed beforehand on an encoding scheme), you can decode the bits and know what phones were in the utterance. For example, if the utterance is "the staircase groaned" I would encode "DH AH S T EH R K EY S G R OW N D", as illustrated at the bottom in Figure 1. We assign bit sequences to each phone unit based on the probability of the unit, meaning our encoding scheme uses as few bits as possible (Shannon 1948; Clover and Thomas 2006; Sanderson 2026). We measure how many bits this optimal phone-based encoding uses.

```
python utils/inspect_example.py 84-121123-0001
```

But, rather than phones, we can decide to tokenise the utterance into another linguistic unit, like syllables. The goal is still to transmit the phonetic content, but we just do it with a different set of units, e.g.\ "DH-AH S-T-EH-R K-EY-S G-R-OW-N-D". Using the optimal syllable-to-bits encoding scheme (now based on the syllable unit probabilities), I again encode the units, send the bits, you decode the bits into syllables, and then you decode the syllables into phones. A different tokenisation, but we've communicated the same phones exactly. Is this more efficient than the phone encoding? Rather than syllables, why not words? Now we have a single unit "S-T-EH-R-K-EY-S" (staircase). Or maybe can communicate even better with multi-word units like "DH-AH-S-T-EH-R-K-EY-S" (the staircase); if in my dataset these two words very often co-occur, surely I could save bits by encoding these together. By now you probably get the picture.

This paper asks which units (phones, syllables, words, units longer than words, something in between) are the most efficient for communicating linguistic content. All experiments are done on transcribed phonetic sequences (obtained from forced alignments) on English data (audiobooks from the LibriSpeech corpus). Why do this comparison? First, the relative efficiencies of these different linguistic tokenisations are not available in the literature. Studies report bitrates like 50 bits-per-second (bps) for phones (Lopes and Aguiar 2010; Flanagan et al. 2014) and 39 bps for syllables (Coupé et al., 2019).^[In defence of the latter, their goal was not to produce *the* bitrate of syllables, but rather to compare information rates across different languages when encoding the same translated content.] A closer reading, however, reveals that even these bitrates are not directly comparable, because the measurement of information rate is not the same and the data differs. (To make a direct comparison, information rates have to be measured consistently and on the same data, as I do here.)

A second reason for this comparison is its link to human language processing, which is likely shaped by pressures for efficient communication (Gibson et al. 2019). I do not want to state this link too strongly, because this paper presents a purely information theoretic analysis ignoring human mental effort, how the mental message is converted into the phonetic surface form in the first place, production and perception effort, and even noise over the communication channel. Nevertheless, characterising the relative information rates of different linguistic units may help inform theories of linguistic representation, including how the mental lexicon is organised (see e.g. Virpioja et al. 2018).

The third motivation is for the zero-resource speech processing community (which is where I come from). With both scientific and engineering aims, this community tries to learn linguistically meaningful units directly from unlabelled speech audio without any transcriptions. Studies have considered the discovery of phone-like units (), syllabic units (), words () and even phrase-like units (). But why should we be interested in particular units rather than others? To be clear: none of this paper deals with speech; all experiments are on transcribed phonetic sequences. But still, as guidance for this community, it is worth knowing whether some units are more efficient than others in an idealised communication setting.

After defining information rate (measured in bits-per-second, bps) below, I first show that all units in fact have the same bitrate when we have a perfect model. This is because different tokenisations of the same input are just different encodings of the same underlying events and should therefore have the same probabilities. But this is only true with a perfect model -- something we never have. I therefore ask the more specific question of what the different information rates look like when using an actual model to estimate the probabilities. I consider two classes of models: class $n$-gram language models and modern transformer-based neural language models.

I train $n$-gram and transformer models on different tokenisations of the same data. Apart from phones, syllables and words, I also apply byte pair encoding (BPE) () at the word-token level, generating multi-word units, like DH-AH-S-T-EH-R-K-EY-S (the staircase). A range of sanity checks at intermediate levels are also performed (like subword BPE merging phones). For the main comparison, all units have a one-to-one correspondents with their constituent phones, e.g. "two" and "too" are both represented as "T-UW". This makes bitrates directly comparable across units since exactly the same content is being transmitted. But other sanity checks are also performed, e.g. operating at the character level.

On the data used here, the best bitrates are achieved with a 20M-parameter transformer operating on words (?? bps). With $n$-grams, adding a small number of word-BPE merges improves results over isolated words. For both model classes, if too many word-merges are added, bitrate gets worse. Under a constrained modelling budget, it therefore seems that a balance needs to be struck between having a small set of shorter-duration units (e.g. phones; best bit rate ??? bps with transformer) vs having a much larger set of longer units (e.g. phrase-like multi-word combinations; ??? bps with transformer when adding 5000 word merges).



## Information rate (bitrate)

The definition of information rate (or bitrate), measured in bits per second, is (Coupé et al. 2019):
$$
\begin{aligned}
    \text{bitrate}
    &= \frac{\text{bits}}{\text{duration}} \\
    &= \frac{\text{\#units}}{\text{duration}} \times \text{bits per unit} \\
\end{aligned}
$$
To know the average bits per unit, we need to decide on an encoding scheme to assign bit sequences to unit types. If we use entropy coding, where the number of bits are chosen based on the probability of the unit, then the bits per unit would be given by the entropy of the distribution over the units (Shannon 1948; Clover and Thomas 2006; Sanderson 2026). But, to do this, we would need to know the true unit probabilities $P(w_t | w_{1:t-1})$.^[I use $w$ here, but note that these units don't need to be words. The whole point is that we will look at different units.] We don't have these. Instead we have a model $P_{\theta}(w_t | w_{1:t-1})$. Under this model, we can then estimate the number of bits on held out data (consisting of $T$ tokens):
$$
\text{bits per unit} = - \frac{1}{T} \sum_{t = 1}^T \log_2 P_{\theta}(w_t | w_{1:t-1})
$$
Formally, this is the estimated cross-entropy rate when we use model $P_{\theta}$ to encode units coming from the true distribution $P$. From this, we can measure the bitrate under the model:
$$
\text{bitrate}
= \frac{\text{bits}}{\text{duration}}
= -\frac{1}{D} \sum_{t = 1}^T \log_2 P_{\theta}(w_t | w_{1:t-1})
= -\frac{1}{D} \log_2 P_{\theta}(w_{1:T})
$$
where $D$ is the duration in seconds of the data.

**Research question.** What are the relative coding efficiencies of different linguistic units (phones, syllables, words, etc.)? Essentially we are asking what $w$ should be to give us low bitrates.

**Other bitrate calculations.** Bitrate has been calculated differently elsewhere. The Zero Resource Speech Challenges, for instance, uses (Dunbar et al. 2019):
$$
    \text{bitrate} = - \frac{ T }{D} \sum_{w} P_{\theta}(w) \log_2 P_{\theta}(w)
$$
Here $P_{\theta}(w)$ is probabilities from a unigram model estimated from unit counts on the evaluation data. Calculating the expected value under this distribution is equivalent to estimating bitrate on the data on which the counts were obtained. Coupé et al. (2019) does something similar, but with a second-order model that takes the single preceding token into account:
$$
    \text{bitrate}
    = -\frac{T}{D} \sum_{w_{t - 1}} \sum_{w_{t}} P_{\theta}(w_{t - 1}, w_t)
        \log_2 \frac{ P_{\theta}(w_{t - 1}, w_t) }{ P_{\theta}(w_{t - 1})}
$$
Both these match my definition if (a) we make a first/second order modelling assumption, and (b) calculate bitrate on the same data used for the unigram/bigram model estimates. This is reasonable for coarse comparisons, but (a) units can presumably be encoded much more efficiently with larger contexts using a higher capacity $P_{\theta}(w)$, and (b) these measurements do not account for generalisation to unseen data.



## But units don't matter: they all have the same ideal bitrate

We now have an appropriate way to measure information rate, though there is still a question of the model $P_{\theta}$. In this section I briefly digress and first show that, under an ideal model, all units will have the same bitrate. But a comparison is still meaningful if the model $P_{\theta}$ is constrained.

The experiments in this paper consistently use phones as the smallest unit from which larger units are built. But for illustration here, let us assume we are using words and considering whether we should rather use larger units. For instance, maybe in your data the word "York" is always preceded by "New". It then seems that we could save bits by having one unit "New York" instead of two separate units "New" and "York", which would each separately take bits. Turns out this isn't true: it doesn't matter how you tokenise the input. The following should be true:
$$
\begin{aligned}
    P(\texttt{New York is beautiful})
    &= P(\texttt{New}) \cdot P( \texttt{York} | \texttt{New}) \cdot P( \texttt{is} | \texttt{New York}) \cdots \\
    &= P(\texttt{New York}) \cdot P( \texttt{is} | \texttt{New York}) \cdots
\end{aligned}
$$
But how does this make sense, given that we should have saved bits by merging "New" and "York"? Turns out that if "York" always Followed "New" (and no other word did), then $P(\texttt{York}|\texttt{New}) = 1$. And so $P(\texttt{New York}) = P(\texttt{New})$. In information theoretic terms: it takes zero extra bits to transmit `York`. So whether you glue it onto `New` or leave it on its own, it costs the same number of bits. More formally, different tokenisations correspond to different random variables defined on the same underlying sample space. I could even split my inputs up into characters, and the result should still be the same as the above: $P(\texttt{N}) \cdot P(\texttt{e} | \texttt{N}) \cdot P(\texttt{w} | \texttt{Ne}) \cdots$.

So, apparently, the question of the relative information rates of different units is silly. They are all the same. But this relies on a strong assumption.

The assumption above is that we know the true probability distribution $P$, the actual distribution governing the real world. But no-one (except the Creator) actually knows these probabilities. So we always have a model $P_{\theta}$. This model will be finite. And this model would be tied to a particular type of unit. And then the above doesn't happen: it now matters how you tokenise your input. If you had a really really powerful model that were really able to perfectly estimate all the probabilities, then yes, the units doesn't matter. But under limited capacity, it does matter which units you use.

**Research question refined.** What are the relative coding efficiencies of different linguistic units *under a limit modelling capacity*?



## Models

I consider two classes of models: $n$-gram language models (Chen and Goodman 1999; Jurafsky and Martin 2026) and transformer-based neural language models (Vaswani et al. 2017).

### Classic *n*-gram models

For the $n$-gram models, I use absolute discounting (Chen and Goodman 1999, \S2.6) in which higher-order $n$-gram language models are interpolated with lower-order models. Discounted mass are used to deal with types that are unseen at training time.

Let's look at the highest order $n$, where we get the probability $P_n(w_t | w_{t - n + 1 : t - 1})$. To simplify the notation, I use $h = w_{t - n + 1 : t - 1}$ to denote the preceding symbols. We then have:
$$
    P_n(w_t | w_{t - n + 1 : t - 1})
    = P_n(w_t | h)
    = \frac{\max \{ c(h, w_t) - d, 0\}}{c(h)} + (1 - \lambda_{h})P_{n - 1}(w_t | h')
$$
where $h' = w_{t - n + 2 : t - 1}$ drops the first symbol from $h$. For these probabilities to sum to one, we need
$$
    1 - \lambda_{h} = \frac{d}{c(h)} N_{1+}(h\cdot)
$$
where $N_{1+}(h\cdot) = |\{w : c(h, w) > 0\}|$ is the number of distinct continuations
observed after $h$.
We therefore get
$$
    P_n(w_t | h) = \frac{ \max \{ c(h, w_t) - d, 0\} } {c(h)} +  \frac{d N_{1+}(h\cdot)} { c(h) } P_{n - 1}(w_t | h')
$$
This recurses down to the unigram case:
$$
    P_1(w_t) = \frac{ \max \{ c(w_t) - d, 0\} } {T} + \frac{Vd}{T} P_0 \quad, \quad P_0 = \frac{1}{V + 1}
$$
The $+1$ in the denominator of $P_0$ reserves probability mass for out-of-vocabulary tokens (I will explain this below).

Note that above I wrote a single discount value $d$ for all orders. These are in fact set separately for each order, following the recommendation of Ney et al.\ (1994):
$$
    d = \frac{n_1}{n_1 + 2n_2}
$$
where $n_1$ and $n_2$ are the number of distinct $n$-grams ocurring exactly once and exactly twice in the training data (at this order). If these counts are zero, a value of $d = 0.75$ is used as fall-back (I did not tune this value). I also clamp the discount to the range $d \in [0.1, 0.9]$ to avoid very strange values.

**Out-of-vocabulary tokens.** When dealing with words and syllables, we have to deal with tokens that are not seen during training. These are all mapped to `<unk>`. But then we back off to a spelling model, paying bits to spell out the token into its constituent phones. So if we have a token "G-R-OW-N-D" (groaned) that has never been seen, it will be decomposed into five tokens "G R OW N D" (with a final end-of-word symbol).

What technically happens is that when we hit `<unk>` in the $P_0$ above, it doesn't use $\frac{1}{V + 1}$ (as it does for all the seen tokens). Instead it backs off to a spelling model $P_\mathrm{spell}(\mathrm{spelling}(w_t))$. This model is itself an absolute discounted $n$-gram using the interpolation as above, but trained on spelled out tokens. (I always use a third-order spelling model.) So, if we wanted to be a bit more formal, I would write
$$
    P_0(w_t) =
    \begin{cases}
    \frac{1}{V + 1} & \text{if\ } w_t \text{\ is seen} \\
    \frac{1}{V + 1} P_\mathrm{spell}(\mathrm{spelling}(w_t)) & \text{if\ } w_t \text{\ is unseen} \\
    \end{cases}
$$
How I visualise this is that $\frac{1}{1 + V}$ mass is allocated to each of the $V$ seen types. For unseen types, the remaining $\frac{1}{1 + V}$ mass is spread out, but non-uniformly based on the probability of their constituent spellings (i.e.\ a type that is easier to spell gets a larger chunk of the $\frac{1}{1 + V}$-`<unk>`-mass). The spelling model ends with an end-of-word symbol (otherwise the probabilities would not be normalised).

**Implementation.** In development experiments, I varied the amount of training data (25%, 50%, 75%, 100%) and the $n$-gram order (from 1 to 5) for all the different tokenisations (below). For most of the units, trigrams worked the best, and I therefore use this throughout. This is then our fixed $n$-gram modelling budget.^[The only exception was phone models, where a five-gram worked better. I give results for all the models with five-grams as well in the Supporting Material.]

### Transformer models

I use a decoder-only autoregressive transformer language models based on a modification of the OPT-125M architecture (Zhang et al. 2022). I reduced the model size so that the body (excluding the output unit-predicting layer) has approximately 20M parameters. This is kept constant regardless of the tokenisation. Each model is a standard causal transformer using stacked blocks with masked multi-head self-attention and a feed-forward layer, pre-layer-norm, ReLU activations, learned positional embeddings, and tied input and output embeddings. The variant I use has 6 layers with 512 hidden dimensions, 8 heads, feedforward networks with 2048 dimensions and a context window of 2048 tokens. This gives all the models the same body of roughly 20M parameters.^[The number of tokens per parameter differ based on the unit, but for phones, this gives 1.69 tokens per parameter. I could probably have gotten away with even smaller models, but I didn't tune this: I picked 20M and moved on with life. But later you see that I do also check bitrates with a much bigger separately trained Qwen () model. This sanity check gives some confidence that the budget choice here is reasonable.] What does differ between the models depending on the unit is the output layer. More detail is given on the different units below, but the output of a phone neural language model only has 39 classes (together with sentence-boundary and pad tokens) while the word model has roughly 58k classes.^[You could argue that I had to balance model sizes taking the (sometimes substantial) output layer into account. The problem is that it is not at all clear how this should be done. Also note that the word and all the word-BPE models have roughly the same number of outputs because of the way the vocabularies are constructed (details to follow).]

**Out-of-vocabulary tokens.** As with the $n$-grams, sometimes we have unseen tokens at test time. We follow a similar spelling-out strategy here: an unseen word is spelled out in its constituent phones. For the transformer models, these spelled-out tokens are also marked to distinguish them from single-phone words, e.g. AA-R-D-W-UH-L-F (aardwolf) would be spelled out as ~AA ~R ~D ~W ~UH ~L ~F, which means that the pronunciation of the word "a" as AA is distinct from `~AA`, the latter being explicitly marked as a word-part. All training words not in the vocabulary are broken up this way. But, during training, I also randomly (2% of the time) break a word up into its constituent phones (whether it is seen or not). This is so that the model sees more broken up words at training time and is then better able to deal with unseen words at test time. The same procedure is followed with spelling out unseen syllables and word-level BPE units into their constituent phones.

**Implementation.** All the neural models are trained with an effective batch size of 40 on a single NVidia RTX3070 (8Gb). I use AdamW optimisation (Loshchilov and Hutter 2017) with a linear warmup for 8% of the steps to a maximum learning rate of $\text{1}\cdot\text{10}^{-3}$ followed by a cosine decay to $\text{2}\cdot\text{10}^{-5}$ over the remaining steps. Models are trained for approximately 20 epochs; the step sizes are changed based on the unit to ensure that the equivalent of roughly 20 passes is made over the encoded data. All these values were chosen based on the original recipe from (Visser et al. 2026). The only change was to raise the maximum learning rate from $\text{5}\cdot\text{10}^{-4}$ to $\text{1}\cdot\text{10}^{-3}$; this slightly improved final performance across all the units (but the trends were always the same). I also experimented with training for more epochs, but this didn't help any of the units. No further development experiments were done.



## Data and units

I perform experiments on the multi-speaker English LibriSpeech corpus of read audiobooks. Models are trained on forced alignments of the 960h training set and evaluated on dev-clean (4.5 hours of speech). The phone-level forced alignments (McAuliffe et al. 2017) uses an ARPABet set without stress markers, giving a set of 39 phones. The goal of this paper is to compare bitrates of different tokenisations of this same data. I compare phones, syllables, words, multi-word combinations obtained from BPE at the word-token level, as well as some intermediate units.

To obtain syllable units from the phone-level forced alignments, a rule-based method is used (Gorman 2013), as was also done in (Visser et al. 2026). Word boundaries are not taken into account, i.e.\ syllables can span words. As noted in the introduction, words are defined here based on phonetic realisations, e.g.\ e.g. "two" and "too" are both represented as the word type "T-UW". I do this because otherwise it isn't possible to directly recover the exact phone sequence from the word tokenisation, breaking the one-to-one correspondence needed for proper bitrate comparisons. The number of tokens and types for phones, syllables and words in the training data are shown in the table below.

| Units | Tokens | Types | Non-singleton types |
|---|---:|---:|---:|
| Phones | 33,464,416 | 39 | 39 |
| Syllables | 13,025,947 | 13,656 | 11,386 |
| Words | 9,403,427 | 84,163 | 56,255 |

```
librispeech_stats.md
```

For the word transformer models, only types occurring more than once in the training data are used in the output vocabulary, giving an output layer of 56k types (last column and row of table). Singletons are treated as unseen (and are therefore spelled out as described above).

Maybe the most efficient units are something larger than words, like phrases or common-multi-word combinations. To test this, I use BPE, but not the standard BPE where subwords are built up by merging commonly co-occurring characters (Gage 1994; Sennrich et al. 2016). (I do include sanity checks using subword BPE below.) Rather, here, multi-word combinations are found by merging at the word-token level. New types like AH-V-DH-AH (of the) are therefore added to the vocabulary. I again constrain the starting vocabulary to words occurring at least twice (58k types), and then add the new merged word types. I consider 200, 1,000 and 5,000 merges. For the transformer models, this increases the output vocabulary by the corresponding merge count. Unseen words are spelled out as phones, as with the normal word model. When doing 5,000 merges, the longest unit is AA-N DH-IY AH-DH-ER S-AY-D AH-V DH-AH (on the other side of the).



## Results

### Main results

The goal of this paper is to see which unit has the best information rate under a constrained modelling budget. For the main comparison, we look at phones, syllables, words and different word-BPE tokenisations (at 200, 1000 and 5000 merges). This gives a total of six units for comparison. We consider two model classes: third-order $n$-gram language model and a transformer language model. The results for both are given in the figure below. Error bars for the transformer model are standard deviations across five seeds.

![Bitrates for different linguistic units per model class. The dark bar gives the best performance per model class. The error bars for the transformer models are bitrate standard deviations over five seeds.](docs/fig/unit_bars_main/unit_bars.pdf){width=85%}

Under both the $n$-gram and transformer bitrate measurements, we see that a unit larger than syllables but smaller than long multi-word bombinations (word-BPE 1,000 or 5,000) gives the lowest bitrates. For the $n$-gram model, this is the word-BPE 200 tokenisation, while for the transfomer, this is word tokenisation. The latter also achieves the best overall bitrate of 28.07 bps.

Although the differences for the $n$-gram models are small, bootstrap confidence estimates (Bisani and Ney) reveal that all pair-wise differences are signigicant at the 99% level, except for the word vs the word-BPE 1,000 models (which makes sense given that their results are both 31.05 bps).^[I use $B = 10,000$ bootstrap samples at the speaker level, i.e.\ 40 speakers are sampled with replacement with the corresponding utterances forming the bootstrap sample.] Pair-wise differences between transformer models are statistically significant in all cases.

### Additional experiments and sanity checks

My goal was to compare the information rates of the linguistic units above, but other (intermediate) units and results are useful for supporting the results and their interpretation.

The first of these is using the more standard form of BPE, where the smallest units (normally characters but here I use phones) are merged to form subword units. Here I apply BPE to the phone units to obtain new types like AH-N (an) and DH-AH (the). I merge to get a vocabulary of 2,000 and disallow merges across word boundaries (as is standard with BPE). The hypothesis would be that the bitrate of this tokenisation would lie somewhere between phones and words (maybe close to syllables), and this is indeed the case: 31.59 bps under the $n$-gram and 28.82$\pm$0.279 bps under the transformer. Presumably, with this approach, we could get operating points anywhere between phones and words by adjusting the vocabulary accordingly.^[With merges close to the word vocabulary (56k), this phonetic subword-BPE tokenisation should match the results with words.]

All of the main results above are performed on phones as the smallest unit. As a sanity check (or maybe an alternative that walks away from speech-based linguistic units), we could have also done the analysis on characters, i.e. words are represented in their orthographic form so that `two` and `too` would now be separate entries in the vocabulary. Results is given in the figure below using transformers (similar 20M-parameter models) to measure bitrate for characters, subword-BPE-merged characters (2,000 merges), ortographic words, and larger units where word tokens are merged with BPE (200, 1,000 and 5,000 merges). 

In this figure, bitrates are lower than those in the figure above, but these are not directly comparable: phones and characters do not have a one-to-one correspondence. Nevertheless, we see the same trend, with word units giving significantly lower bitrates than the other units when based on characters.

![Bitrates for different character-based units using transformers for bitrate calculation.](docs/fig/unit_bars_char/unit_bars.pdf){width=85%}

Finally, I wanted to gauge how close we are getting to the true bitrate; given the discussion in Section ??, this clearly isn't possible because we will never have the real model. Neverhtless, we can consider bitrates when training data and model size are dramatically scaled. I cannot train models of this size myself, but I can use available open-weight models. This means we are also tied to this model's units (so we cannot compare different tokenisations), but this is acceptable given that our aim now is not unit comparison. For this experiment, I use two Qwen2.5 models (Qwen Team, 2024), a 1.5B- and a 3B-parameter model. These models are trained on roughly 18T tokens of text. They use a byte-based BPE tokeniser with an output vocabulary of about 150k types. Because Qwen's training domain and punctuation is quite different from the normalised LibriSpeech text that I evaluate on, the models are conditioned on a lowercase unpunctuated primer drawn from train-clean-100, prepended to every evaluation utterance. (This lowers bitrate for both models.) With this setup, Qwen2.5-1.5B achieves a bitrate of 25.90 bps, while Qwen2.5-3B achieves 25.26 bps. These are directly comparable to the character-based result in Figure ??, where the best overall bitrate is 27.48. This shows that my models are in the right ballpark.

## Discussion, limitations and future work



As model size and amount of data scales, we should expect the different units to give similar bitrates. But the point of this study was to do a comparison under the same budget, and this serves as a reasonable sanity check.


## Discussion

We are not encoding the actual "message", which probably has much fewer bits, but also not the actual speech (speaker, emotion, channel), which probably would cost way more bits.

Just because encoding is more efficient, it doesn't necessarily mean generation is going to be better (it almost certainly does not). It is fine to spend a large number of bits on long sequences that occur once or twice, but if you accidentally generated these during inference in the wrong context, the penalty could be very unintelligible output.



## Conclusion and future work

I do not claim that words are the most efficient units (especially given that the concept of what a word is is contested, and also quite language dependent). What I would claim is that the most efficient unit is something bigger than phones or syllables and smaller than long multi-word combinations.

Other languages, especially languages where the definition of a word is different.



## Limitations

Other languages

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

I wrote the review for this PhD in its entirety, and then used Claude to fix spelling and grammar mistakes and highlight unclear sentences. I then reviewed the entire document again.


## Acknowledgements

- Nicol Visser - 3Blue1Brown, code
- Benjamin van Niekerk
- Simon Malan for the alignments



## References

"Information-theoretic analysis of tokenisation and text predictability in South African languages," in *SACAIR*, 2026.

Byung-Doh Oh, William Schuler, “The Impact of Token Granularity on the Predictive Power of Language Model Surprisal”, https://arxiv.org/abs/2412.11940. 

Virpioja et al. (2018), “Using Statistical Models of Morphology in the Search for Optimal Units of Representation in the Human Mental Lexicon.”

Gibson et al. (2019), “How Efficiency Shapes Human Language.”

Dunbar et al. (2019), "The Zero Resource Speech Challenge 2019: TTS without T"



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



## More details on units

**Phones.** Phone-level forced aligments for the data were produced using Montreal Forced Aligner (McAuliffe et al. 2017) using an ARPABet phone set. Stress markers are removed, leaving a set of 39 phones.

**Syllables.** As in (Visser et al. 2026), a syllable tokenisation is obtained from the phone sequences using a rule-based method (Gorman 2013). Word boundaries are not taken into account, i.e.\ syllables can span over words. When encountering an out-of-vocabulary syllable, the model backs off to a "spelled out" version of the syllable, where "spelling" here means using the constituent phones. (This spelling model is used in both the $n$-gram and neural models.)

**Words.** For the word case, out-of-vocabulary words are spelled out as characters. For the neural model, only words that occur at least twice in the training data are used in the output vocabulary, resulting in 58k classes. Words occurring only once or not at all in the training data is therefore spelled out.

**Word-BPE.** Maybe the most efficient units are something larger than words, like phrases or common-multi-word combinations. To test this, I use BPE, but not the standard BPE where subwords are built up by merging commonly co-occurring characters (Gage 1994; Sennrich et al. 2016). (Below I do do a sanity check using standard subword-BPE.) Rather, here, multi-word combinations are found by merging at the word-token level. New types like `of the` are therefore added to the vocabulary. I again constrain the starting vocabulary to words occurring at least twice (58k types), and then add the new merged word types. I consider 200, 1000 and 5000 merges. Unseen words are spelled out as characters, as with the plain word model.



## Word-BPE

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



### Notes on development experiments on single units

Bitrates for syllable units as the amount of data is varied:

| fraction | order 1 | order 2 | order 3 | order 4 | order 5 | OPT-20M |
|---|---:|---:|---:|---:|---:|---:|
| 25% | 46.22 | 35.24 | 32.98 | 33.06 | 33.25 | - |
| 50% | 46.21 | 34.91 | 32.28 | 32.30 | 32.51 | - |
| 75% | 46.20 | 34.78 | 31.94 | 31.91 | 32.13 | - |
| 100% | 46.20 | 34.68 | 31.68 | 31.61 | 31.82 | 28.41 |

This trend was the same for all models, except phones, for which a 5-gram was the best (38.45 for 4-gram vs 34.93 bps for 5-gram when using 100% of the data). But I wanted a fixed order for all the models, so I picked trigrams. (And even if I did use a 5-gram just for phones and a 3-gram for all the others, the conclusions would still be the same. The problem with this is that would allow for a changing budget, which violates our research question. In the Appendix I give the order 5 results for all the models.)
