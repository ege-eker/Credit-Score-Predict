import {Dispatch, SetStateAction} from "react";

type DropdownInputProps<E extends Record<number, string>> = {
	label: string,
	setState: Dispatch<SetStateAction<E[keyof E]>>,
	options: E
}

export function DropdownInput<E extends Record<number, string>> ({
	setState,
	label,
	options
}: DropdownInputProps<E>){

	let keys = Object.keys(options)
	let optionEls = keys.map((key) => {
		let val = options[key as keyof E]!
		return <option key={key} value={val.toString()} >{key}</option>
	})

	return (
		<div className={"flex flex-col gap-1.5 w-full h-full"}>
			<p>{label}:</p>
			<select
				className={"p-2 ml-1 border-gray-400 border shadow-xl rounded-md w-full h-full bg-white"}
				onChange={(event) => {
					const element = event.target as HTMLSelectElement
					setState(element.value as E[keyof E])
				}}
			>
				{optionEls}
			</select>
		</div>
	);
}