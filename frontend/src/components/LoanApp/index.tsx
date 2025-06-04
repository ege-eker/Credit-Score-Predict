"use client"

import {useState} from "react";
import InputStage from "@/components/LoanApp/InputStage";
import {z} from "zod";
import {Button} from "@/components/Button";


export const ApiRequestBodySchema = z.object({
	params: z.object({
		income: z.number().int(),
		age: z.number().int(),
		employment_length: z.number().int(),
		loan_amount: z.number().int(),
		loan_intent: z.enum(["VENTURE", "PERSONAL", "MEDICAL", "HOMEIMPROVEMENT", "EDUCATION", "DEBTCONSOLIDATION"]),
		home_ownership: z.enum(["OWN", "RENT", "MORTGAGE"]),
		default_on_file: z.boolean()
	}),
	model: z.enum(["gb", "lgb", "nn", "rf", "svm", "xgb"])
})

export type Stage = "configuring" | "processing" | "completed"

export default function LoanApp() {
	const [data, setData] = useState<z.infer<typeof ApiRequestBodySchema> | null>(null)
	const [stage, setStage] = useState<Stage>("configuring")
	const [result, setResult] = useState<null | string>(null)



	return <>
		{
			stage === "configuring" || stage === "processing" ? <InputStage setData={setData} setStage={setStage} stage={stage} data={data} setResult={setResult}/>
				: <div className={"bg-gray-100 rounded-md h-max p-8 flex flex-col justify-between items-center gap-10 text-black"}>
					<div>
						<p className={"text-4xl"}>Credit Score Predictor</p>
					</div>
					<div>
						<p className={"text-xl"}>
							Result: {result}
						</p>
					</div>
					<div className={"w-full"}>
						<Button onClick={()=>{
							setStage("configuring")
						}}
						label={"Go Back"}></Button>
					</div>
				</div>
		}
	</>
}
