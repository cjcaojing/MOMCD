for /l %%i in (1,1,10) do (
	java -cp bin LinLogLayout 2 football_m8.el football_m8_result_%%i.emb
)