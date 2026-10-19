<div align="center">
  <img
    src="https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/kpnn2_logo.png"
    alt="kpnn2 logo"
    height="72"
    align="middle"
  >
  <img
    src="https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/kpnn2_wordmark.png"
    alt="kpnn2"
    height="72"
    align="middle"
  >
</div>

---

[![ci](https://img.shields.io/github/actions/workflow/status/Thomas-Rauter/kpnn2/ci.yml?branch=main&label=ci&logo=github&labelColor=555)](https://github.com/Thomas-Rauter/kpnn2/actions/workflows/ci.yml)
[![codecov](https://img.shields.io/codecov/c/github/Thomas-Rauter/kpnn2?logo=codecov&labelColor=555)](https://app.codecov.io/gh/Thomas-Rauter/kpnn2)
[![PyPI](https://img.shields.io/pypi/v/kpnn2?labelColor=555&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGZpbGwtcnVsZT0iZXZlbm9kZCIgaGVpZ2h0PSIzNjguNTY4IiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbWl0ZXJsaW1pdD0iMS41IiB2aWV3Qm94PSIxMzguOTk4IDExMi4wNzkgMzE3LjMxMCAzNjguNTY4IiB3aWR0aD0iMzE3LjMxMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3Mi41OSAtNjMuMjA5KSI%2BPHBhdGggZD0ibTY3My40MSAyMzYuMDE2LTY0LjI3MSA4LjI1MnYyODEuMzU3aDY0LjI3NnoiIGZpbGw9IiNmZmNhMWUiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjExIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAyMzY1IC0uMzcyNzUgLjIyMTE1IC4wODA5MyAtNDIzLjA4NiA0NTcuNDQ4KSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIuMzUiIHRyYW5zZm9ybT0ibWF0cml4KDMuOTMzNTkgLTEuNDM4MzcgMCAuNTMwODQgLTIyNjYuNDMgMTA4Ny45NCkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2U9IiNkMWUzZjIiIHN0cm9rZS13aWR0aD0iLjYyIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjk2MDQgLS43MTY4NSAuMjIxNiAuMDc5MjcgLTExMTguNTUgNjM5LjMzOSkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMzc3NWE4IiBzdHJva2U9IiNkMWUzZjIiIHN0cm9rZS13aWR0aD0iMS4xNSIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wMjI4MSAtLjM3NCAwIDEuMDU2OTUgLTQzMC43NzMgMjE0LjE3OCkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS4xNyIgdHJhbnNmb3JtPSJtYXRyaXgoLS45NzQ5OSAtLjM0OTI0IDAgMS4wNTY5NSA3ODYuMTk0IDE5OS4yMDgpIi8%2BPHBhdGggZD0ibTYwOS4xMzkgMjQ0LjI2OGg2NC4yNzZ2MjgxLjM1N2gtNjQuMjc2eiIgZmlsbD0iI2VmZWVlYSIgc3Ryb2tlPSIjZDhkOGQ4IiBzdHJva2Utd2lkdGg9IjEuNDQiIHRyYW5zZm9ybT0ibWF0cml4KC0uOTc0OTkgLS4zNTY1MiAwIC4yNjg4NSA3ODYuMTk0IDYxOC4zNTUpIi8%2BPGcgc3Ryb2tlPSIjZDFlM2YyIj48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2Utd2lkdGg9IjEuNDQiIHRyYW5zZm9ybT0ibWF0cml4KC0uOTY4MzQgLS4zNTQwOSAwIC41MzA3NyA3MTkuNDgzIDQyNy41KSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMzNzc1YTgiIHN0cm9rZS13aWR0aD0iMS4yIiB0cmFuc2Zvcm09Im1hdHJpeCguOTM1NTQgLS4zNDIxIDAgMS4wNTY5NSAtMzExLjg5MiAxNzAuNDkyKSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMzNzc1YTgiIHN0cm9rZS13aWR0aD0iMS40MyIgdHJhbnNmb3JtPSJtYXRyaXgoLjk3NDIgLS4zNTYyMyAwIC41MzA4NCAtNDYzLjc0NCA0MjguNzYxKSIvPjxwYXRoIGQ9Im02Ny41NzUgMzkzLjE2MSA2Mi4xMjEgMjIuNDY1IDE4OC43MDgtNjguMjk5bS0xMjUuMTY1LTI5LjE0MSAxMjQuNzMyLTQ1LjYwMiIgZmlsbD0ibm9uZSIvPjwvZz48cGF0aCBkPSJtMzE4LjQwNCAzNDcuMzI3IDYzLjkzOS0yMy4yMDkiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2Q3YzViMiIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMyZjY0OTAiIHN0cm9rZT0iI2QxZTNmMiIgc3Ryb2tlLXdpZHRoPSIxLjE2IiB0cmFuc2Zvcm09Im1hdHJpeCguOTY3ODggLS4zNTI0NCAuMjIxMTUgLjA4MDkzIC01NzYuMTY4IDUxMy41ODMpIi8%2BPGNpcmNsZSBjeD0iNjM3LjUxNyIgY3k9IjI2MC4wMDEiIGZpbGw9IiNmZmYiIHI9IjE1LjcxIiB0cmFuc2Zvcm09Im1hdHJpeCguNzgyNiAtLjQwMjQgLjA1NDk0IC44NjE0IC0yOTUuMzYzIDMwNC45MzQpIi8%2BPHBhdGggZD0ibTE5NS43ODYgMTk4LjEyNSA2MS42OTYgMjIuMTI2IiBmaWxsPSJub25lIiBzdHJva2U9IiNkMWUzZjIiLz48cGF0aCBkPSJtNjczLjQxNSAyNDQuMjY4aC02NC4yNzZsLjAxOCAyODIuNDA1IDY0LjI1OC0xLjA0OHoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjM3IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAyMjgxIC0uMzc0IDAgLjUyODQzIC00MzAuNzczIDQ5MS45ODMpIi8%2BPHBhdGggZD0ibTY3My40MTUgMjQ0LjI2OGgtNjQuMjc2bC4wMDEgMjgxLjc1OCA2NC4yNzUtLjQwMXoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjQ4IiB0cmFuc2Zvcm09Im1hdHJpeCguOTM1NTQgLS4zNDIxIDAgLjUyNzQzIC0zMTEuODkyIDQ0OC44MjIpIi8%2BPGNpcmNsZSBjeD0iNjM3LjUxNyIgY3k9IjI2MC4wMDEiIGZpbGw9IiNmZWZkZmQiIHI9IjE1LjcxIiB0cmFuc2Zvcm09Im1hdHJpeCguNzcwNzQgLS4zOTYzIC4wNTE1NiAuODA4MzIgLTIwNS45MTYgNTA5LjQxMSkiLz48cGF0aCBkPSJtMTkyLjQxMiA0NjguMDU5IDEyNi4wMjgtNDUuOTc3IiBmaWxsPSJub25lIiBzdHJva2U9IiNkN2M1YjIiLz48L2c%2BPC9zdmc%2B)](https://pypi.org/project/kpnn2/)
[![pypi since](https://img.shields.io/badge/pypi-since%20September%202026-blue?labelColor=555&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGZpbGwtcnVsZT0iZXZlbm9kZCIgaGVpZ2h0PSIzNjguNTY4IiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbWl0ZXJsaW1pdD0iMS41IiB2aWV3Qm94PSIxMzguOTk4IDExMi4wNzkgMzE3LjMxMCAzNjguNTY4IiB3aWR0aD0iMzE3LjMxMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3Mi41OSAtNjMuMjA5KSI%2BPHBhdGggZD0ibTY3My40MSAyMzYuMDE2LTY0LjI3MSA4LjI1MnYyODEuMzU3aDY0LjI3NnoiIGZpbGw9IiNmZmNhMWUiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjExIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAyMzY1IC0uMzcyNzUgLjIyMTE1IC4wODA5MyAtNDIzLjA4NiA0NTcuNDQ4KSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIuMzUiIHRyYW5zZm9ybT0ibWF0cml4KDMuOTMzNTkgLTEuNDM4MzcgMCAuNTMwODQgLTIyNjYuNDMgMTA4Ny45NCkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2U9IiNkMWUzZjIiIHN0cm9rZS13aWR0aD0iLjYyIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjk2MDQgLS43MTY4NSAuMjIxNiAuMDc5MjcgLTExMTguNTUgNjM5LjMzOSkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMzc3NWE4IiBzdHJva2U9IiNkMWUzZjIiIHN0cm9rZS13aWR0aD0iMS4xNSIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wMjI4MSAtLjM3NCAwIDEuMDU2OTUgLTQzMC43NzMgMjE0LjE3OCkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS4xNyIgdHJhbnNmb3JtPSJtYXRyaXgoLS45NzQ5OSAtLjM0OTI0IDAgMS4wNTY5NSA3ODYuMTk0IDE5OS4yMDgpIi8%2BPHBhdGggZD0ibTYwOS4xMzkgMjQ0LjI2OGg2NC4yNzZ2MjgxLjM1N2gtNjQuMjc2eiIgZmlsbD0iI2VmZWVlYSIgc3Ryb2tlPSIjZDhkOGQ4IiBzdHJva2Utd2lkdGg9IjEuNDQiIHRyYW5zZm9ybT0ibWF0cml4KC0uOTc0OTkgLS4zNTY1MiAwIC4yNjg4NSA3ODYuMTk0IDYxOC4zNTUpIi8%2BPGcgc3Ryb2tlPSIjZDFlM2YyIj48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2Utd2lkdGg9IjEuNDQiIHRyYW5zZm9ybT0ibWF0cml4KC0uOTY4MzQgLS4zNTQwOSAwIC41MzA3NyA3MTkuNDgzIDQyNy41KSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMzNzc1YTgiIHN0cm9rZS13aWR0aD0iMS4yIiB0cmFuc2Zvcm09Im1hdHJpeCguOTM1NTQgLS4zNDIxIDAgMS4wNTY5NSAtMzExLjg5MiAxNzAuNDkyKSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMzNzc1YTgiIHN0cm9rZS13aWR0aD0iMS40MyIgdHJhbnNmb3JtPSJtYXRyaXgoLjk3NDIgLS4zNTYyMyAwIC41MzA4NCAtNDYzLjc0NCA0MjguNzYxKSIvPjxwYXRoIGQ9Im02Ny41NzUgMzkzLjE2MSA2Mi4xMjEgMjIuNDY1IDE4OC43MDgtNjguMjk5bS0xMjUuMTY1LTI5LjE0MSAxMjQuNzMyLTQ1LjYwMiIgZmlsbD0ibm9uZSIvPjwvZz48cGF0aCBkPSJtMzE4LjQwNCAzNDcuMzI3IDYzLjkzOS0yMy4yMDkiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2Q3YzViMiIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMyZjY0OTAiIHN0cm9rZT0iI2QxZTNmMiIgc3Ryb2tlLXdpZHRoPSIxLjE2IiB0cmFuc2Zvcm09Im1hdHJpeCguOTY3ODggLS4zNTI0NCAuMjIxMTUgLjA4MDkzIC01NzYuMTY4IDUxMy41ODMpIi8%2BPGNpcmNsZSBjeD0iNjM3LjUxNyIgY3k9IjI2MC4wMDEiIGZpbGw9IiNmZmYiIHI9IjE1LjcxIiB0cmFuc2Zvcm09Im1hdHJpeCguNzgyNiAtLjQwMjQgLjA1NDk0IC44NjE0IC0yOTUuMzYzIDMwNC45MzQpIi8%2BPHBhdGggZD0ibTE5NS43ODYgMTk4LjEyNSA2MS42OTYgMjIuMTI2IiBmaWxsPSJub25lIiBzdHJva2U9IiNkMWUzZjIiLz48cGF0aCBkPSJtNjczLjQxNSAyNDQuMjY4aC02NC4yNzZsLjAxOCAyODIuNDA1IDY0LjI1OC0xLjA0OHoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjM3IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAyMjgxIC0uMzc0IDAgLjUyODQzIC00MzAuNzczIDQ5MS45ODMpIi8%2BPHBhdGggZD0ibTY3My40MTUgMjQ0LjI2OGgtNjQuMjc2bC4wMDEgMjgxLjc1OCA2NC4yNzUtLjQwMXoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjQ4IiB0cmFuc2Zvcm09Im1hdHJpeCguOTM1NTQgLS4zNDIxIDAgLjUyNzQzIC0zMTEuODkyIDQ0OC44MjIpIi8%2BPGNpcmNsZSBjeD0iNjM3LjUxNyIgY3k9IjI2MC4wMDEiIGZpbGw9IiNmZWZkZmQiIHI9IjE1LjcxIiB0cmFuc2Zvcm09Im1hdHJpeCguNzcwNzQgLS4zOTYzIC4wNTE1NiAuODA4MzIgLTIwNS45MTYgNTA5LjQxMSkiLz48cGF0aCBkPSJtMTkyLjQxMiA0NjguMDU5IDEyNi4wMjgtNDUuOTc3IiBmaWxsPSJub25lIiBzdHJva2U9IiNkN2M1YjIiLz48L2c%2BPC9zdmc%2B)](https://pypi.org/project/kpnn2/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue?labelColor=555&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48IURPQ1RZUEUgc3ZnIFBVQkxJQyAiLS8vVzNDLy9EVEQgU1ZHIDEuMS8vRU4iICJodHRwOi8vd3d3LnczLm9yZy9HcmFwaGljcy9TVkcvMS4xL0RURC9zdmcxMS5kdGQiPjxzdmcgdmVyc2lvbj0iMS4xIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOmNjPSJodHRwOi8vd2ViLnJlc291cmNlLm9yZy9jYy8iIHhtbG5zOnJkZj0iaHR0cDovL3d3dy53My5vcmcvMTk5OS8wMi8yMi1yZGYtc3ludGF4LW5zIyIgeG1sbnM6c3ZnPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgeD0iMHB4IiB5PSIwcHgiIHdpZHRoPSIxMTBweCIgaGVpZ2h0PSIxMTBweCIgdmlld0JveD0iMC4yMSAtMC4wNzcgMTEwIDExMCIgZW5hYmxlLWJhY2tncm91bmQ9Im5ldyAwLjIxIC0wLjA3NyAxMTAgMTEwIiB4bWw6c3BhY2U9InByZXNlcnZlIj48bGluZWFyR3JhZGllbnQgaWQ9IlNWR0lEXzFfIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9IjYzLjgxNTkiIHkxPSI1Ni42ODI5IiB4Mj0iMTE4LjQ5MzQiIHkyPSIxLjgyMjUiIGdyYWRpZW50VHJhbnNmb3JtPSJtYXRyaXgoMSAwIDAgLTEgLTUzLjI5NzQgNjYuNDMyMSkiPiA8c3RvcCBvZmZzZXQ9IjAiIHN0eWxlPSJzdG9wLWNvbG9yOiMzODdFQjgiLz4gPHN0b3Agb2Zmc2V0PSIxIiBzdHlsZT0ic3RvcC1jb2xvcjojMzY2OTk0Ii8%2BPC9saW5lYXJHcmFkaWVudD48cGF0aCBmaWxsPSJ1cmwoI1NWR0lEXzFfKSIgZD0iTTU1LjAyMy0wLjA3N2MtMjUuOTcxLDAtMjYuMjUsMTAuMDgxLTI2LjI1LDEyLjE1NmMwLDMuMTQ4LDAsMTIuNTk0LDAsMTIuNTk0aDI2Ljc1djMuNzgxIGMwLDAtMjcuODUyLDAtMzcuMzc1LDBjLTcuOTQ5LDAtMTcuOTM4LDQuODMzLTE3LjkzOCwyNi4yNWMwLDE5LjY3Myw3Ljc5MiwyNy4yODEsMTUuNjU2LDI3LjI4MWMyLjMzNSwwLDkuMzQ0LDAsOS4zNDQsMCBzMC05Ljc2NSwwLTEzLjEyNWMwLTUuNDkxLDIuNzIxLTE1LjY1NiwxNS40MDYtMTUuNjU2YzE1LjkxLDAsMTkuOTcxLDAsMjYuNTMxLDBjMy45MDIsMCwxNC45MDYtMS42OTYsMTQuOTA2LTE0LjQwNiBjMC0xMy40NTIsMC0xNy44OSwwLTI0LjIxOUM4Mi4wNTQsMTEuNDI2LDgxLjUxNS0wLjA3Nyw1NS4wMjMtMC4wNzd6IE00MC4yNzMsOC4zOTJjMi42NjIsMCw0LjgxMywyLjE1LDQuODEzLDQuODEzIGMwLDIuNjYxLTIuMTUxLDQuODEzLTQuODEzLDQuODEzcy00LjgxMy0yLjE1MS00LjgxMy00LjgxM0MzNS40NiwxMC41NDIsMzcuNjExLDguMzkyLDQwLjI3Myw4LjM5MnoiLz48bGluZWFyR3JhZGllbnQgaWQ9IlNWR0lEXzJfIiBncmFkaWVudFVuaXRzPSJ1c2VyU3BhY2VPblVzZSIgeDE9Ijk3LjA0NDQiIHkxPSIyMS42MzIxIiB4Mj0iMTU1LjY2NjUiIHkyPSItMzQuNTMwOCIgZ3JhZGllbnRUcmFuc2Zvcm09Im1hdHJpeCgxIDAgMCAtMSAtNTMuMjk3NCA2Ni40MzIxKSI%2BIDxzdG9wIG9mZnNldD0iMCIgc3R5bGU9InN0b3AtY29sb3I6I0ZGRTA1MiIvPiA8c3RvcCBvZmZzZXQ9IjEiIHN0eWxlPSJzdG9wLWNvbG9yOiNGRkMzMzEiLz48L2xpbmVhckdyYWRpZW50PjxwYXRoIGZpbGw9InVybCgjU1ZHSURfMl8pIiBkPSJNNTUuMzk3LDEwOS45MjNjMjUuOTU5LDAsMjYuMjgyLTEwLjI3MSwyNi4yODItMTIuMTU2YzAtMy4xNDgsMC0xMi41OTQsMC0xMi41OTRINTQuODk3di0zLjc4MSBjMCwwLDI4LjAzMiwwLDM3LjM3NSwwYzguMDA5LDAsMTcuOTM4LTQuOTU0LDE3LjkzOC0yNi4yNWMwLTIzLjMyMi0xMC41MzgtMjcuMjgxLTE1LjY1Ni0yNy4yODFjLTIuMzM2LDAtOS4zNDQsMC05LjM0NCwwIHMwLDEwLjIxNiwwLDEzLjEyNWMwLDUuNDkxLTIuNjMxLDE1LjY1Ni0xNS40MDYsMTUuNjU2Yy0xNS45MSwwLTE5LjQ3NiwwLTI2LjUzMiwwYy0zLjg5MiwwLTE0LjkwNiwxLjg5Ni0xNC45MDYsMTQuNDA2IGMwLDE0LjQ3NSwwLDE4LjI2NSwwLDI0LjIxOUMyOC4zNjYsMTAwLjQ5NywzMS41NjIsMTA5LjkyMyw1NS4zOTcsMTA5LjkyM3ogTTcwLjE0OCwxMDEuNDU0Yy0yLjY2MiwwLTQuODEzLTIuMTUxLTQuODEzLTQuODEzIHMyLjE1LTQuODEzLDQuODEzLTQuODEzYzIuNjYxLDAsNC44MTMsMi4xNTEsNC44MTMsNC44MTNTNzIuODA5LDEwMS40NTQsNzAuMTQ4LDEwMS40NTR6Ii8%2BPC9zdmc%2B)](https://pypi.org/project/kpnn2/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.1-blue?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyPI - License](https://img.shields.io/pypi/l/kpnn2?labelColor=555)](https://pypi.org/project/kpnn2/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/kpnn2?labelColor=555&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGZpbGwtcnVsZT0iZXZlbm9kZCIgaGVpZ2h0PSIzNjguNTY4IiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBzdHJva2UtbWl0ZXJsaW1pdD0iMS41IiB2aWV3Qm94PSIxMzguOTk4IDExMi4wNzkgMzE3LjMxMCAzNjguNTY4IiB3aWR0aD0iMzE3LjMxMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3Mi41OSAtNjMuMjA5KSI%2BPHBhdGggZD0ibTY3My40MSAyMzYuMDE2LTY0LjI3MSA4LjI1MnYyODEuMzU3aDY0LjI3NnoiIGZpbGw9IiNmZmNhMWUiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjExIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAyMzY1IC0uMzcyNzUgLjIyMTE1IC4wODA5MyAtNDIzLjA4NiA0NTcuNDQ4KSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIuMzUiIHRyYW5zZm9ybT0ibWF0cml4KDMuOTMzNTkgLTEuNDM4MzcgMCAuNTMwODQgLTIyNjYuNDMgMTA4Ny45NCkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2U9IiNkMWUzZjIiIHN0cm9rZS13aWR0aD0iLjYyIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjk2MDQgLS43MTY4NSAuMjIxNiAuMDc5MjcgLTExMTguNTUgNjM5LjMzOSkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMzc3NWE4IiBzdHJva2U9IiNkMWUzZjIiIHN0cm9rZS13aWR0aD0iMS4xNSIgdHJhbnNmb3JtPSJtYXRyaXgoMS4wMjI4MSAtLjM3NCAwIDEuMDU2OTUgLTQzMC43NzMgMjE0LjE3OCkiLz48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2U9IiNmZmYiIHN0cm9rZS13aWR0aD0iMS4xNyIgdHJhbnNmb3JtPSJtYXRyaXgoLS45NzQ5OSAtLjM0OTI0IDAgMS4wNTY5NSA3ODYuMTk0IDE5OS4yMDgpIi8%2BPHBhdGggZD0ibTYwOS4xMzkgMjQ0LjI2OGg2NC4yNzZ2MjgxLjM1N2gtNjQuMjc2eiIgZmlsbD0iI2VmZWVlYSIgc3Ryb2tlPSIjZDhkOGQ4IiBzdHJva2Utd2lkdGg9IjEuNDQiIHRyYW5zZm9ybT0ibWF0cml4KC0uOTc0OTkgLS4zNTY1MiAwIC4yNjg4NSA3ODYuMTk0IDYxOC4zNTUpIi8%2BPGcgc3Ryb2tlPSIjZDFlM2YyIj48cGF0aCBkPSJtNjA5LjEzOSAyNDQuMjY4aDY0LjI3NnYyODEuMzU3aC02NC4yNzZ6IiBmaWxsPSIjMmY2NDkwIiBzdHJva2Utd2lkdGg9IjEuNDQiIHRyYW5zZm9ybT0ibWF0cml4KC0uOTY4MzQgLS4zNTQwOSAwIC41MzA3NyA3MTkuNDgzIDQyNy41KSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMzNzc1YTgiIHN0cm9rZS13aWR0aD0iMS4yIiB0cmFuc2Zvcm09Im1hdHJpeCguOTM1NTQgLS4zNDIxIDAgMS4wNTY5NSAtMzExLjg5MiAxNzAuNDkyKSIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMzNzc1YTgiIHN0cm9rZS13aWR0aD0iMS40MyIgdHJhbnNmb3JtPSJtYXRyaXgoLjk3NDIgLS4zNTYyMyAwIC41MzA4NCAtNDYzLjc0NCA0MjguNzYxKSIvPjxwYXRoIGQ9Im02Ny41NzUgMzkzLjE2MSA2Mi4xMjEgMjIuNDY1IDE4OC43MDgtNjguMjk5bS0xMjUuMTY1LTI5LjE0MSAxMjQuNzMyLTQ1LjYwMiIgZmlsbD0ibm9uZSIvPjwvZz48cGF0aCBkPSJtMzE4LjQwNCAzNDcuMzI3IDYzLjkzOS0yMy4yMDkiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2Q3YzViMiIvPjxwYXRoIGQ9Im02MDkuMTM5IDI0NC4yNjhoNjQuMjc2djI4MS4zNTdoLTY0LjI3NnoiIGZpbGw9IiMyZjY0OTAiIHN0cm9rZT0iI2QxZTNmMiIgc3Ryb2tlLXdpZHRoPSIxLjE2IiB0cmFuc2Zvcm09Im1hdHJpeCguOTY3ODggLS4zNTI0NCAuMjIxMTUgLjA4MDkzIC01NzYuMTY4IDUxMy41ODMpIi8%2BPGNpcmNsZSBjeD0iNjM3LjUxNyIgY3k9IjI2MC4wMDEiIGZpbGw9IiNmZmYiIHI9IjE1LjcxIiB0cmFuc2Zvcm09Im1hdHJpeCguNzgyNiAtLjQwMjQgLjA1NDk0IC44NjE0IC0yOTUuMzYzIDMwNC45MzQpIi8%2BPHBhdGggZD0ibTE5NS43ODYgMTk4LjEyNSA2MS42OTYgMjIuMTI2IiBmaWxsPSJub25lIiBzdHJva2U9IiNkMWUzZjIiLz48cGF0aCBkPSJtNjczLjQxNSAyNDQuMjY4aC02NC4yNzZsLjAxOCAyODIuNDA1IDY0LjI1OC0xLjA0OHoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjM3IiB0cmFuc2Zvcm09Im1hdHJpeCgxLjAyMjgxIC0uMzc0IDAgLjUyODQzIC00MzAuNzczIDQ5MS45ODMpIi8%2BPHBhdGggZD0ibTY3My40MTUgMjQ0LjI2OGgtNjQuMjc2bC4wMDEgMjgxLjc1OCA2NC4yNzUtLjQwMXoiIGZpbGw9IiNmZmQyNDEiIHN0cm9rZT0iI2Q3YzViMiIgc3Ryb2tlLXdpZHRoPSIxLjQ4IiB0cmFuc2Zvcm09Im1hdHJpeCguOTM1NTQgLS4zNDIxIDAgLjUyNzQzIC0zMTEuODkyIDQ0OC44MjIpIi8%2BPGNpcmNsZSBjeD0iNjM3LjUxNyIgY3k9IjI2MC4wMDEiIGZpbGw9IiNmZWZkZmQiIHI9IjE1LjcxIiB0cmFuc2Zvcm09Im1hdHJpeCguNzcwNzQgLS4zOTYzIC4wNTE1NiAuODA4MzIgLTIwNS45MTYgNTA5LjQxMSkiLz48cGF0aCBkPSJtMTkyLjQxMiA0NjguMDU5IDEyNi4wMjgtNDUuOTc3IiBmaWxsPSJub25lIiBzdHJva2U9IiNkN2M1YjIiLz48L2c%2BPC9zdmc%2B)](https://pypi.org/project/kpnn2/)

Build interpretable neural networks in PyTorch from a graph of
named entities, with checks that every attribution is reported
under the right name.

## What it is

The hidden neurons of a standard neural network have no meaning
outside the model. An attribution method can say which neurons a
prediction relied on, but not what those neurons stand for.

A knowledge-primed neural network (KPNN) is built from a graph of
known entities, such as proteins and the interactions between
them, as Figure 1 shows. Every neuron belongs to one entity, and
two neurons are connected only where the graph has an edge
([Fortelny and Bock, 2020](https://doi.org/10.1186/s13059-020-02100-5)).
The attributions of a KPNN therefore name proteins, not anonymous
neurons.

![Left: a network of known interactions, from a receptor through kinases and transcription factors to genes. Right: the same wiring as a neural network, with input nodes at the bottom, hidden nodes in between, and one output node at the top.](https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/KPNNs_explained.png)

**Figure 1.** From a network of known interactions (left) to a
knowledge-primed neural network with the same wiring (right).

`kpnn2` builds such networks in PyTorch from an edgelist, a table
of edges between named entities. Any domain whose entities have
names and known relationships works the same way. When each sample
is its own graph, use a graph neural network instead; see
[Why not a GNN?](https://thomas-rauter.github.io/kpnn2/latest/supported/#why-not-a-gnn).

## Why it exists

> **"A Scientist's Nightmare: Software Problem Leads to Five
> Retractions"**
>
> In 2006 a structural biology lab retracted five papers,
> including papers in Science. An in-house program had swapped
> two columns of data, which inverted the protein structures the
> papers reported. Nothing crashed and the results looked
> plausible. Other labs had built on the wrong structures for
> five years before another group's structure disagreed.
> ([Miller, 2006](https://doi.org/10.1126/science.314.5807.1856))

A knowledge-primed network has the same weak spot. Inside PyTorch
a protein is only column 17 of a table of numbers. If your code
puts it in the wrong column, or the order changes later, the model
still trains and the loss looks normal, but the attributions land
on the wrong proteins. `kpnn2` aims to prevent that.

Three reasons to use it:

- **Checked for correctness.** `kpnn2` builds the wiring from the
  edgelist, so you write no masks or indices by hand. It matches
  input columns by name, refuses a checkpoint with different
  wiring, and rejects an attribution tensor of the wrong width. Its
  tests check that only edges in the edgelist can affect a
  prediction;
  [How we test](https://thomas-rauter.github.io/kpnn2/latest/how_we_test/)
  also lists what they do not prove.
- **Less code, less to get wrong.** In the side-by-side example in
  [Why not custom PyTorch?](https://thomas-rauter.github.io/kpnn2/latest/why_kpnn2/#why-not-custom-pytorch),
  one network takes 36 lines with `kpnn2` and 162 in hand-written
  PyTorch, and the hand-written version does not yet align columns,
  check checkpoints, or label attributions. Every line you do not
  write is one where a column cannot slip.
- **Built on, not rebuilt.** `kpnn2` provides tested layouts and
  layers for skip edges, feedback loops, and attention limited to
  the graph's edges. You start from them instead of designing and
  validating your own, and spend your time on the new question.

[**Why kpnn2**](https://thomas-rauter.github.io/kpnn2/latest/why_kpnn2/)
makes the full case and lists every check.

## How it works

Each row of the edgelist is one edge, from a `source` node to a
`target` node. `kpnn2` parses the edgelist, you build the network
in PyTorch from the result and train it, and `kpnn2` labels the
attributions with node names (Figure 2). `kpnn2` provides building
blocks, not a finished model: activations, losses, the training
loop, and the attribution method stay your own PyTorch code.

![Left: a seven-row edgelist of generic proteins, and a data table with one row per input protein plus an outcome row. Middle: the same graph as a network, in which protein_a and protein_b feed protein_e, protein_c and protein_d feed protein_f, both feed outcome, and a dashed edge from protein_d to outcome passes under the hidden layer. Right: a table of attributions with one row per hidden protein, dark for protein_e and pale for protein_f in every sample.](https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/kpnn2_overview.png)

**Figure 2.** The edgelist, the data, the network, and the
attributions all use the same node names. Here the attributions
show a model that relies on `protein_e` in every sample and hardly
on `protein_f`.

By default, `parse_layered()` places each node in a layer by the
longest path that reaches it from an input. An edge that spans
more than one layer, such as the dashed edge in Figure 2, is a
skip edge. In the edgelist it is one more row, and in the network
it gets its own learned weight from the earlier layer: no relay
neurons are added in between, and it is not a residual
connection. A graph with feedback loops has no layers.
`parse_adjacency()` keeps every node in one vector, and your
`forward()` applies the same connections for as many steps as you
choose.

[Layered vs. unlayered](https://thomas-rauter.github.io/kpnn2/latest/layered_vs_adjacency/)
and the other How it works pages explain how the network is
built, and
[Attributions by node name](https://thomas-rauter.github.io/kpnn2/latest/map-node-attributions/)
explains how the attributions get their names.

## Installation

Requires Python 3.10 or later.

```bash
pip install kpnn2
```

## Quick start

This example builds a model from a small simulated graph, trains
it, and reads its
[attributions](https://thomas-rauter.github.io/kpnn2/latest/concepts/#attribution)
by node name. Because the data are simulated, you know in advance
which node the model should rely on, and you can check that the
attributions agree. The last step uses Captum
(`pip install captum`), which `kpnn2` does not depend on.

### 1. Write the graph as an edgelist

An
[edgelist](https://thomas-rauter.github.io/kpnn2/latest/concepts/#edgelist)
has one row per edge, from `source` to `target`. `parse_layered()`
sorts the nodes into layers.

```python
import pandas as pd
import torch
from torch import nn

import kpnn2

torch.manual_seed(42)

edgelist = pd.DataFrame(
    [
        ("input_signal_1", "hidden_signal"),
        ("input_signal_2", "hidden_signal"),
        ("input_noise_1", "hidden_noise"),
        ("input_noise_2", "hidden_noise"),
        ("hidden_signal", "output"),
        ("hidden_noise", "output"),
    ],
    columns=["source", "target"],
)
spec = kpnn2.parse_layered(edgelist)
for layer in spec.layer_nodes:
    print(layer)
```

```text
('input_noise_1', 'input_noise_2', 'input_signal_1', 'input_signal_2')
('hidden_noise', 'hidden_signal')
('output',)
```

In Figure 3, the graph splits into a signal branch and a noise
branch that meet only at `output`.

![The quick-start graph from left to right. In a box labeled signal branch, input_signal_1 and input_signal_2 feed hidden_signal. In a box labeled noise branch, input_noise_1 and input_noise_2 feed hidden_noise. Both hidden nodes feed output.](https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/quick_start_graph.png)

**Figure 3.** The quick-start graph: inputs on the left, the hidden
layer in the middle, and `output` on the right.

Step 3 computes the labels from the two signal inputs only. The
signal branch therefore carries the information the model needs to
make its prediction, so a model that learned the task should rely
on `hidden_signal` and give it a high attribution. The two noise
inputs are random numbers unrelated to the labels. The noise branch
carries nothing the prediction can use, so `hidden_noise` should
get a very low attribution.

### 2. Build the model in PyTorch

Each [hop](https://thomas-rauter.github.io/kpnn2/latest/concepts/#hop)
— the edges entering one layer — becomes one `PackedLinear`, which
stores one weight per edge; `PackedLinear.from_hop()` builds it
from the hop. Without
[skip edges](https://thomas-rauter.github.io/kpnn2/latest/concepts/#skip-edge),
each hop reads only the layer before it, so `nn.Sequential` is
enough. With them, a hop also reads earlier layers, and
`gather_hop_inputs()` assembles its input; see
[Edges that skip layers](https://thomas-rauter.github.io/kpnn2/latest/skip-edges/).
`identity=spec.fingerprint` makes a checkpoint from a different
graph refuse to load.

```python
hop_0, hop_1 = spec.hops
model = nn.Sequential(
    kpnn2.PackedLinear.from_hop(
        hop_0,
        identity=spec.fingerprint,
    ),
    nn.Tanh(),
    kpnn2.PackedLinear.from_hop(
        hop_1,
        identity=spec.fingerprint,
    ),
)
```

### 3. Train on data matched by name

`align_inputs()` puts your feature columns in the order the model
expects, by name, so a table in any column order lines up.

```python
input_names = [
    "input_signal_1",
    "input_signal_2",
    "input_noise_1",
    "input_noise_2",
]
features = pd.DataFrame(
    torch.randn(
        200,
        4,
    ).numpy(),
    columns=input_names,
)
labels = features["input_signal_1"] + features["input_signal_2"] > 0

col = kpnn2.align_inputs(
    features.columns,
    spec,
)
x = torch.as_tensor(
    features.to_numpy()[:, col],
    dtype=torch.float32,
)
y = torch.as_tensor(
    labels.to_numpy(),
    dtype=torch.float32,
).unsqueeze(1)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.05,
)
loss_fn = nn.BCEWithLogitsLoss()
for _ in range(200):
    optimizer.zero_grad()
    loss = loss_fn(
        model(x),
        y,
    )
    loss.backward()
    optimizer.step()
```

### 4. Read attributions by node name

Run any attribution method on the trained model, then label its
output with `map_node_attributions()`. Here Captum's
`LayerConductance` computes attributions for the hidden layer, and
the mean absolute attribution per node summarizes them.

```python
from captum.attr import LayerConductance

conductance = LayerConductance(
    model,
    model[0],
)
attributions = kpnn2.map_node_attributions(
    conductance.attribute(x),
    spec,
    hop_output=hop_0,
)
print(abs(attributions).mean("observation").to_pandas().round(2))
```

```text
node
hidden_noise     0.39
hidden_signal    4.90
dtype: float32
```

Figure 4 plots these scores: the trained model relies on
`hidden_signal` and hardly on `hidden_noise`, as the labels
require.

![Horizontal bar chart of the mean absolute attribution per hidden node. The bar for hidden_signal is long, the bar for hidden_noise is short.](https://raw.githubusercontent.com/Thomas-Rauter/kpnn2/main/docs/figures/quick_start_attributions.png)

**Figure 4.** Mean absolute attribution per hidden node in the
trained model, from the run above.

The
[Feedforward example](https://thomas-rauter.github.io/kpnn2/latest/feedforward-example/)
goes further, with a held-out test set, input-level attributions,
and a control that moves the signal to the other branch.

## Next steps

- [**Feedforward example**](https://thomas-rauter.github.io/kpnn2/latest/feedforward-example/)
  for a full tutorial, from edgelist to node-level interpretation
- [**Supported architectures**](https://thomas-rauter.github.io/kpnn2/latest/supported/)
  for cyclic, recurrent, and attention-based models
- [**How we test**](https://thomas-rauter.github.io/kpnn2/latest/how_we_test/)
  for the tests that pin scientific claims, including a
  reproduction of
  [Fortelny and Bock (2020)](https://thomas-rauter.github.io/kpnn2/latest/literature/fortelny-bock-2020/)
- [**API reference**](https://thomas-rauter.github.io/kpnn2/latest/reference/api/)
  for every public name

## Citation

If you use `kpnn2` in research, please cite the software.
Citation metadata is available in
[`CITATION.cff`](https://github.com/Thomas-Rauter/kpnn2/blob/main/CITATION.cff).

## License

This project is licensed under the MIT License. See the
[LICENSE file on GitHub](https://github.com/Thomas-Rauter/kpnn2/blob/main/LICENSE)
for details.
