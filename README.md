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

Turn a named edgelist into sparsely connected PyTorch layers
you assemble yourself.

[**Why kpnn2**](docs/why_kpnn2.md) explains what knowledge-primed
neural networks are and what `kpnn2` adds to plain PyTorch.

## Core workflow

An **edgelist** is a table of directed connections: each row links
a `source` node to a `target` node. For example:

| source | target |
|--------|--------|
| A | H |
| B | H |
| H | C |

The **graph** is what that table describes: the named nodes, and the
directed edges between them. Edgelist and graph are the same object in
two forms — the table you pass in, and the structure it encodes. These
docs use "graph" in that sense throughout: the prior wiring that becomes
the architecture. It never means PyTorch's autograd graph, and never a
graph as *data* in the GNN sense (see [Why not a GNN?](docs/supported.md#why-not-a-gnn)).
[**Concepts**](docs/concepts.md) defines the rest of the vocabulary.

1. Define a model architecture as an edgelist with named `source`
   and `target` [nodes](docs/concepts.md#node).
2. Parse it. For a
   [directed acyclic graph](docs/concepts.md#dag) (DAG) you want
   as layers, call `parse_layered()`. It returns a `LayeredSpec`
   with one packed [hop](docs/concepts.md#hop) per layer — a hop
   being everything entering one layer. For cycles, or for one
   shared [state vector](docs/concepts.md#state-vector) over all
   nodes, call `parse_adjacency()`. It returns an `AdjacencySpec`
   whose edges are [packed indices](docs/concepts.md#packed-indices).
   A DAG is valid for both, so the layout is your choice.
3. Write an `nn.Module` with one `PackedLinear` per hop in
   `spec.hops`, and feed each one `gather_hop_inputs(saved, hop)`.
   [Skip edges](docs/concepts.md#skip-edge), whose endpoints are
   more than one layer apart, already sit inside those hops, so
   there is nothing extra to call. `MaskedLinear(hop.to_mask())`
   is the dense hatch for small graphs.
4. Align feature names with `align_inputs()`, then index the
   host matrix.
5. Train with ordinary PyTorch.
6. Optionally run Captum (or another method) yourself, then label a
   layer tensor with `map_node_attributions()` (returns xarray).
7. Optionally fold those scores with `aggregate_node_attributions()`.
8. A checkpoint is `spec.to_dict()` plus `state_dict`, not
   weights alone.

The snippet below is a minimal run of steps 1–4, using the
edgelist from the table above. Column order in the input table
does not matter: `align_inputs()` matches names and returns a
column index. Skip edges are
omitted here; [**Skip edges**](docs/skip-edges.ipynb) works them
through. [**Why not custom PyTorch?**](docs/why_kpnn2.md#why-not-custom-pytorch)
sets this hop loop against the equivalent module written by hand.
A full walkthrough, including training and attribution, is in
[**Feedforward example**](docs/feedforward-example.ipynb).

```python
import pandas as pd
import torch
from torch import nn

import kpnn2

edgelist = pd.DataFrame(
    {
        "source": ["A", "B", "H"],
        "target": ["H", "H", "C"],
    }
)
spec = kpnn2.parse_layered(edgelist)


class Net(nn.Module):
    def __init__(self, spec: kpnn2.LayeredSpec):
        super().__init__()
        self.lin0 = kpnn2.PackedLinear(
            spec.hops[0].source_index,
            spec.hops[0].target_index,
            spec.hops[0].out_features,
            spec.hops[0].in_features,
            identity=spec.fingerprint,
        )
        self.lin1 = kpnn2.PackedLinear(
            spec.hops[1].source_index,
            spec.hops[1].target_index,
            spec.hops[1].out_features,
            spec.hops[1].in_features,
            identity=spec.fingerprint,
        )
        self.acts = nn.ModuleList(
            [nn.ReLU() for _ in spec.hops]
        )

    def forward(self, x):
        h = self.acts[0](self.lin0(x))
        return self.lin1(h)


model = Net(spec)
features = pd.DataFrame({"B": [0.2, 0.4], "A": [0.1, 0.3]})
col = kpnn2.align_inputs(features.columns, spec)
x = torch.as_tensor(
    features.to_numpy()[:, col],
    dtype=torch.float32,
)
y = model(x)
# Continue training with ordinary PyTorch.
```

## Installation

Requires Python 3.10 or later.

```bash
pip install kpnn2
```

## Start here

If you are new to the package, start with a tutorial:

- [**Installation**](docs/installation.md) for package setup
- [**Supported architectures**](docs/supported.md) for which
  architecture families this package covers
- [**Feedforward example**](docs/feedforward-example.ipynb) for a
  full end-to-end feedforward network

## Additional examples

- [**Cyclic graph example**](docs/cyclic-graph-example.ipynb) for
  a graph with a feedback loop: `parse_adjacency()`, one shared
  `MaskedLinear` over the state vector, train, and interpret
  named nodes (`parse_layered` still requires a DAG)
- [**Time-series example**](docs/time-series-example.ipynb) for a
  sequence `x_t`: the same shared `MaskedLinear`, with a new
  input written at each time, and a self-loop so named nodes
  carry state (`nn.RNN` cannot take an edgelist)
- [**Transformer example**](docs/transformer-example.ipynb) for
  `PackedMultiheadAttention` on those packed indices, as a
  prior-gated encoder you write yourself

The other pages are not second examples:

- [**Why kpnn2**](docs/why_kpnn2.md) for knowledge-primed neural
  networks, a side-by-side with hand-written PyTorch, and what
  `kpnn2` checks
- [**Concepts**](docs/concepts.md) for the vocabulary these docs
  use: graph, edgelist, spec, hop, skip edge, live edge, and the
  rest, in the order the pipeline uses them
- [**Layered vs. Adjacency**](docs/layered_vs_adjacency.md) for
  how the two parsers differ and when to pick one
- [**Skip edges**](docs/skip-edges.ipynb) for edges that jump a
  layer, and why they need no separate mechanism
- [**Mapping attributions**](docs/map-node-attributions.ipynb) for
  labeling layer tensors with node names
- [**PackedLinear**](docs/packed_linear.md) when `n` is large on
  an `AdjacencySpec`
- [**How we test**](docs/how_we_test.md) for the tests that pin
  wiring and interpretation claims
- [**API reference**](docs/reference/api.md) for function- and object-level
  documentation

## Citation

If you use `kpnn2` in research, please cite the software.
Citation metadata is available in
[`CITATION.cff`](https://github.com/Thomas-Rauter/kpnn2/blob/main/CITATION.cff).

## License

This project is licensed under the MIT License. See the
[LICENSE file on GitHub](https://github.com/Thomas-Rauter/kpnn2/blob/main/LICENSE)
for details.
