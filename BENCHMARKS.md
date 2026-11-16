# lvim-fuzzy — benchmarks

lvim-fuzzy is the engine behind `lvim-picker` and `lvim-cmp`: lvim-fuzzy-core, reached from Neovim
through a thin C ABI. It is measured at two levels.

## The engine

lvim-fuzzy-core against frizbee (the matcher behind blink.cmp), and against nucleo, skim's
fuzzy-matcher and sublime_fuzzy: one million paths, real source paths, lines of code and Cyrillic
text, on every core and on one thread — in
[lvim-fuzzy-core](https://github.com/lvim-tech/lvim-fuzzy-core#benchmark).

## The picker

lvim-picker against fzf-lua, snacks.picker, mini.pick and telescope, each as a whole plugin with its
default settings: the picker opens on the same command (`cat <corpus>`), the whole list loads, a
query is typed as keys and timed until that plugin's result for it is ready. Milliseconds, lower is
better; AMD Ryzen 9 3900XT (24 threads). The method, the notes and the harness are in
[lvim-picker](https://github.com/lvim-tech/lvim-picker#speed) (`bench/run.sh`).

**files** — 1 000 000 paths under `$HOME`

| plugin | load | `hel` | `lvim` | `config` | `initlua` | mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **lvim-picker** | **249** | **20.9** | **14.2** | **9.8** | **10.7** | **13.9** |
| fzf-lua | 321 | 160.0 | 104.7 | 66.0 | 53.0 | 96.0 |
| mini.pick | 2 871 | 612.7 | 422.0 | 238.7 | 250.9 | 381.1 |
| snacks.picker | 940 | 1 886 | 1 086 | 1 088 | 1 190 | 1 313 |
| telescope | 4 614 | 34 351 | 22 388 | 10 784 | 9 873 | 19 349 |

**grep** — 300 000 ripgrep lines

| plugin | load | `err` | `function` | `return` | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| **lvim-picker** | **110** | **16.8** | **13.3** | **17.1** | **15.7** |
| fzf-lua | 140 | 74.8 | 22.6 | 65.7 | 54.4 |
| mini.pick | 837 | 292.2 | 117.2 | 194.2 | 201.2 |
| snacks.picker | 296 | 639.2 | 237.1 | 333.6 | 403.3 |
| telescope | 1 235 | 10 727 | 2 503 | 9 251 | 7 494 |

### How many times faster lvim-picker answers a query

| than | files | grep |
| --- | ---: | ---: |
| fzf-lua | 6.9× | 3.5× |
| mini.pick | 27× | 13× |
| snacks.picker | 94× | 26× |
| telescope | 1 392× | 477× |
