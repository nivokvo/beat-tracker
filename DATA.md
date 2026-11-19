# Данные и лицензии (для отчёта Qildo)

Правило Qildo: только данные с коммерческой лицензией (CC BY и т. п.) или свои записи. Без NC, без «без лицензии».
Чужих обученных весов нет: модель и постобработка обучены и написаны с нуля.

## Обучение

| Что | Для чего | Лицензия | Ссылка |
|---|---|---|---|
| Lakh MIDI Dataset (lmd_full) | Синтетические песни: MIDI → аудио своим рендером, доли и сильные доли точно из темп-карты MIDI. Свои аугментации: темп, тембры, «вступление одним инструментом», свинг, свободный темп (rubato), отдельный набор 3/4 и 6/8 | CC BY 4.0 | https://colinraffel.com/projects/lmd/ |
| FluidR3_GM.sf2 (Frank Wen) | Саундфонт для рендера | MIT | Debian `fluid-soundfont-gm`, файл `data/raw/sf_fluidr3/LICENSE.txt` |
| MuseScore_General.sf2 (S. Christian Collins) | Второй саундфонт | MIT | https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General_License.md |
| Slakh2100-redux (Manilow et al.) | Тот же Lakh, отрендеренный профессиональными сэмпл-инструментами; берём только микс и MIDI (`data_prep/fetch_slakh.py`, архив читается потоком) | CC BY 4.0 | https://zenodo.org/records/4599666 |
| Groove MIDI Dataset (Magenta, Google) | Живые барабанщики под клик (13 ч грувов: рок, фанк, джаз, шаффл), смешанные с партиями Lakh, подогнанными под их доли (`data_prep/render_groove.py`) | CC BY 4.0 | https://magenta.tensorflow.org/datasets/groove |
| GuitarSet (игроки 00–04 — обучение, 05 — тест) | Реальная гитара, разметка долей и тактов | CC BY 4.0 | https://zenodo.org/records/3371780 |
| incompetech.com, Кевин Маклауд (~1400 треков) | Реальные песни с BPM автора: псевдоразметка своей моделью, октава темпа сверяется с BPM автора. 150 треков отложены как судья и в обучение не идут | CC BY 4.0 | https://incompetech.com/music/royalty-free/licenses/ |
| Free Music Archive (FMA), только рок и поп под Attribution / CC0 / Public Domain Mark (2475 треков, список с лицензией каждого — `data/raw/fma/selected.csv`) | Настоящие записи рока и попа с живыми барабанами: псевдоразметка двумя своими моделями, берём только треки, где они согласны (967 треков) | CC BY 1.0–4.0, CC0, PDM — у каждого трека своя; NC, SA, ND и «Sound Recording Common Law» исключены | https://github.com/mdeff/fma |
| ccMixter, только Attribution | Реальные песни для псевдоразметки | CC BY 3.0 / 4.0, ссылка у каждого трека в `data/raw/ccmixter/meta.jsonl` | https://ccmixter.org |

FluidSynth (LGPL-2.1) используется только как программа для рендера при подготовке данных, в решение не входит.

## Только проверка (в обучение не входят)

| Что | Срез | Лицензия | Ссылка |
|---|---|---|---|
| KRAISLER (KAIST): скрипка + фортепиано вживую, ручная разметка долей | свободный темп | CC BY 4.0 | https://zenodo.org/records/21082251 |
| Schubert Winterreise Dataset: голос + фортепиано, такты размечены вручную | свободный темп, 3/4 | CC BY 3.0 | https://zenodo.org/records/10839767 |
| Uruguayan candombe drumming: живые барабаны, доли и сильные доли вручную | живые барабаны | CC BY 4.0 | https://zenodo.org/records/6533068 |
| Groove MIDI, test-сплит (барабанщики, которых не было в обучении) | живые барабаны, рок/поп, свинг | CC BY 4.0 | см. выше |

## Что смотрели и не взяли
- Harmonix Set: разметка CC BY 4.0, но аудио не распространяется (только ссылки на YouTube).
- Osu2MIR: аудио из пользовательских карт osu!, прав на коммерческое использование нет.
- MTG-Jamendo, Dagstuhl ChoirSet, Saraga: NC / NC-SA.
- ASAP: аудио из MAESTRO (NC-SA).
- «Beat This! Spectrograms»: спектрограммы сделаны из аудио под копирайтом.
- MusicNet (CC BY 4.0 у записи, но лицензии отдельных записей не указаны, разметка выровнена DTW) — отложено.
- FSL10K: у каждого лупа своя лицензия Freesound, много NC; сетки долей нет.
- GTZAN, Ballroom, Beatles/Isophonics, RWC, SMC, Hainsworth: аудио защищено или без лицензии.
- GeneralUser GS (саундфонт): автор сам не уверен в происхождении части сэмплов.
- JSIR (джаз): CC BY-SA — share-alike, не берём, чтобы не спорить о лицензии весов.
- ccMixter: в ноябре 2026 сайт недоступен (истёк сертификат), новые треки не качали.
- Чужие обученные веса (madmom, Beat This!, BeatNet): запрещены правилами. Из Beat This! взята только идея архитектуры и растяжка по темпу как аугментация.
